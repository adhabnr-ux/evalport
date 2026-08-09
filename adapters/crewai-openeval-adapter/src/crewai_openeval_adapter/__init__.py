"""CrewAI <-> EvalPort adapter.

Standalone converter between CrewAI crew/task evaluation results and the
EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside CrewAI
itself: a native-support proposal has been open on crewAIInc/crewAI#6711
since July 2026 with no maintainer engagement. Rather than block on that,
this package follows the same playbook that worked for AutoGen
(https://github.com/adhabnr-ux/evalport/tree/main/adapters/autogen-openeval-adapter):
it works against CrewAI's public Task/TaskOutput/Crew shapes (objects or
dicts) from the outside, so you get EvalPort import/export today without
needing anything merged into CrewAI's core.

Design credit: the to_openeval()/from_openeval() surface mirrors the mapping
proposed in https://github.com/adhabnr-ux/evalport/issues/5 and discussed
with @Bryan-eng-lng on crewAIInc/crewAI#6711.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    CrewAI's Task/TaskOutput classes and JSON-loaded eval output both show
    up in the wild (and CrewAI's own object shape has changed across
    versions), so every accessor in this module goes through here rather
    than assuming one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tool_names(tools: Any) -> List[str]:
    """Normalize a CrewAI tools list (strings, or Tool objects with `.name`) to plain strings."""
    if not tools:
        return []
    names = []
    for t in tools:
        name = t if isinstance(t, str) else _get(t, "name", None)
        if name:
            names.append(str(name))
    return names


def _task_payload(task: Any, index: int) -> Dict[str, Any]:
    """Normalize a single CrewAI task/task-output into an EvalPort TestCase dict."""
    task_id = _get(task, "id") or _get(task, "task_id") or f"tc_{index}"
    # A pre-run Task exposes `description`; a post-run TaskOutput also
    # exposes `description` (carried over from the Task it came from).
    description = _get(task, "description") or _get(task, "input") or ""
    expected_output = _get(task, "expected_output")
    tools = _tool_names(_get(task, "tools") or _get(_get(task, "agent") or {}, "tools"))
    agent_name = _get(task, "agent")
    agent_name = _get(agent_name, "role", agent_name) if agent_name is not None else None

    graders: List[str] = []
    if expected_output:
        graders.append("gr_output_match")
    if tools:
        graders.append("gr_tool_selection")
    if not graders:
        graders = ["gr_output_match"]

    tc: Dict[str, Any] = {
        "id": str(task_id),
        "input": description,
        "graders": graders,
    }
    if expected_output is not None:
        tc["expected_output"] = str(expected_output)
    if tools:
        tc["expected_tools"] = tools
    if agent_name:
        tc["metadata"] = {"crewai_agent": str(agent_name)}
    return tc


def to_openeval(crew_result: Any, grader_type: str = "llm_judge") -> Dict[str, Any]:
    """Export a CrewAI crew/task result to an EvalPort-shaped suite (dict).

    `crew_result` may be any object or dict exposing a `tasks` or
    `tasks_output` sequence (this matches CrewAI's `CrewOutput.tasks_output`,
    a plain `Crew.tasks` list, or plain JSON-loaded eval output), plus an
    optional `id`/`run_id` — no direct CrewAI import is required.

    `grader_type` selects the output-quality grader: "llm_judge" (default)
    or "exact_match". CrewAI's `expected_output` is conventionally a
    natural-language description of the desired result ("a 3-bullet
    summary"), not a literal string to match character-for-character, so
    `llm_judge` is the sane default here — the opposite default from the
    AutoGen adapter, which usually deals with more literal outputs.

    When a task has `tools` (directly, or via `task.agent.tools`), an
    additional `gr_tool_selection` grader is attached so tool-selection
    accuracy can be checked independently of output text — this is scored
    by whichever EvalPort runner executes the suite, by comparing
    `expected_tools` against the tools actually called.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    tasks = _get(crew_result, "tasks_output") or _get(crew_result, "tasks") or []
    run_id = _get(crew_result, "id") or _get(crew_result, "run_id") or "crewai_run"

    test_cases = [_task_payload(t, i) for i, t in enumerate(tasks)]

    graders: List[Dict[str, Any]] = []
    if any("gr_output_match" in tc["graders"] for tc in test_cases):
        if grader_type == "exact_match":
            graders.append({"id": "gr_output_match", "type": "exact_match", "params": {"ignore_case": True}})
        else:
            graders.append({
                "id": "gr_output_match",
                "type": "llm_judge",
                "params": {
                    "model": "gpt-4o",
                    "prompt": (
                        "Expected outcome: {expected}\nActual agent output: {output}\n"
                        "Does the output satisfy the expected outcome? "
                        'Return JSON: {"score": 0.0-1.0, "reason": "..."}'
                    ),
                },
            })
    if any("gr_tool_selection" in tc["graders"] for tc in test_cases):
        graders.append({
            "id": "gr_tool_selection",
            "type": "custom",
            "description": "CrewAI tool-selection check",
            "params": {"handler": "crewai:tools_subset"},
        })

    return {
        "version": OPENEVAL_VERSION,
        "id": f"crewai_eval_{run_id}",
        "name": f"CrewAI eval run {run_id}",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "crewai"}},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of CrewAI-shaped task dicts.

    Returns plain dicts (description, expected_output, tools) rather than
    a CrewAI `Task` instance, since constructing a real `Task` also
    requires an `agent` this module has no way to know about. Pass the
    fields straight into your own `Task(**{**task, "agent": my_agent})`
    call, or map them explicitly.
    """
    tasks: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        tasks.append(
            {
                "id": tc.get("id"),
                "description": tc.get("input"),
                "expected_output": tc.get("expected_output"),
                "tools": tc.get("expected_tools") or [],
            }
        )
    return tasks
