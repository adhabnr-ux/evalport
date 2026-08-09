"""CrewAI <-> EvalPort converters.

CrewAI evaluates crews by running a set of Tasks (each with a description,
an optional expected_output, and a list of tools an agent is allowed to
call) and inspecting the resulting TaskOutput objects. This module maps
that shape onto EvalPort's TestCase/Grader/ResultSet documents so CrewAI
eval definitions and run results are portable to any other EvalPort-aware
tool (and vice versa).

    from openeval.converters_crewai import from_crewai, crewai_result_to_result_set

    suite = from_crewai({"tasks": [...]})          # CrewAI tasks -> EvalSuite
    result_set = crewai_result_to_result_set(       # CrewAI run -> ResultSet
        crew_result, suite, run_id="run_1"
    )
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .types import OPENEVAL_VERSION


def from_crewai(crew_tasks: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a CrewAI task definition list into an EvalPort EvalSuite.

    `crew_tasks` is expected to look like:
        {
            "tasks": [
                {
                    "id": "task_1",                       # optional
                    "description": "Summarize the report", # -> input
                    "expected_output": "A 3-bullet summary",# optional
                    "tools": ["search_tool", "file_tool"],  # optional
                    "agent": "researcher",                  # optional, kept in metadata
                },
                ...
            ]
        }

    Each task becomes a TestCase. A task with `expected_output` gets a
    `llm_judge` grader (free-form text can't be exact-matched); a task
    with `tools` also gets a `contains`-style `expected_tools` field so
    tool-selection accuracy is checkable independently of output text.
    """
    tasks: List[Dict[str, Any]] = crew_tasks.get("tasks", [])

    graders: List[Dict[str, Any]] = []
    test_cases: List[Dict[str, Any]] = []

    for i, task in enumerate(tasks):
        tc_id = task.get("id", f"tc_{i}")
        description = task.get("description", "")
        expected_output = task.get("expected_output")
        tools = task.get("tools") or []

        tc_graders: List[str] = []

        if expected_output:
            gid = f"gr_{i}_output"
            graders.append({
                "id": gid,
                "type": "llm_judge",
                "description": "CrewAI expected_output match",
                "params": {
                    "model": "gpt-4o",
                    "prompt": (
                        "Expected outcome: {expected}\nActual agent output: {output}\n"
                        "Does the output satisfy the expected outcome? "
                        "Return JSON: {\"score\": 0.0-1.0, \"reason\": \"...\"}"
                    ),
                },
            })
            tc_graders.append(gid)

        if tools:
            gid = f"gr_{i}_tools"
            graders.append({
                "id": gid,
                "type": "custom",
                "description": "CrewAI tool-selection check",
                "params": {"handler": "crewai:tools_subset"},
            })
            tc_graders.append(gid)

        if not tc_graders:
            gid = f"gr_{i}_default"
            graders.append({"id": gid, "type": "human", "params": {"instructions": "Review manually."}})
            tc_graders.append(gid)

        tc: Dict[str, Any] = {
            "id": tc_id,
            "input": description,
            "graders": tc_graders,
        }
        if expected_output:
            tc["expected_output"] = expected_output
        if tools:
            tc["expected_tools"] = tools
        metadata: Dict[str, Any] = {}
        if task.get("agent"):
            metadata["crewai_agent"] = task["agent"]
        if metadata:
            tc["metadata"] = metadata

        test_cases.append(tc)

    return {
        "version": OPENEVAL_VERSION,
        "id": "suite_crewai_import",
        "name": "Imported from CrewAI",
        "graders": graders,
        "test_cases": test_cases,
        "metadata": {"openeval": {"source": "crewai"}},
    }


def crewai_result_to_result_set(
    crew_result: Dict[str, Any],
    suite: Dict[str, Any],
    run_id: str,
    runner_name: str = "evalport-sdk",
    runner_version: str = "1.0.0",
) -> Dict[str, Any]:
    """Convert a CrewAI crew execution result into an EvalPort ResultSet.

    `crew_result` is expected to look like:
        {
            "tasks": [
                {
                    "id": "task_1",             # must match the TestCase id from from_crewai()
                    "output": "...",             # the agent's actual output text
                    "tools_called": ["search_tool"],
                    "duration_ms": 842,
                },
                ...
            ]
        }

    `suite` is the EvalSuite dict produced by `from_crewai` (or an
    equivalent hand-written one) — used to look up expected_output /
    expected_tools per test case so we can compute pass/fail without
    re-running an LLM judge here (that's left to the caller/runner; this
    converter just gets the two documents into a comparable shape).
    """
    tc_by_id = {tc["id"]: tc for tc in suite.get("test_cases", [])}

    results: List[Dict[str, Any]] = []
    for task in crew_result.get("tasks", []):
        tc_id = task.get("id")
        tc = tc_by_id.get(tc_id, {})
        actual_output = task.get("output", "")
        tools_called = task.get("tools_called", [])
        expected_tools = tc.get("expected_tools", [])

        grader_results: List[Dict[str, Any]] = []
        for gid in tc.get("graders", []):
            if gid.endswith("_tools"):
                passed = set(expected_tools).issubset(set(tools_called)) if expected_tools else True
                grader_results.append({
                    "grader_id": gid,
                    "type": "custom",
                    "score": 1.0 if passed else 0.0,
                    "passed": passed,
                    "reason": f"expected {expected_tools}, got {tools_called}",
                })
            else:
                # Output-quality graders (llm_judge/human) need an actual
                # runner to score — record as unscored so downstream
                # tooling knows to run the judge rather than silently
                # treating it as a pass.
                grader_results.append({
                    "grader_id": gid,
                    "type": "llm_judge",
                    "score": None,
                    "passed": False,
                    "reason": "not scored by converter — run through an EvalPort runner",
                })

        results.append({
            "test_case_id": tc_id,
            "passed": all(gr["passed"] for gr in grader_results) if grader_results else False,
            "grader_results": grader_results,
            "actual_output": actual_output,
            "duration_ms": task.get("duration_ms"),
            "metadata": {"tools_called": tools_called} if tools_called else {},
        })

    now = datetime.now(timezone.utc).isoformat()
    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite.get("id", "suite_crewai_import"),
        "suite_version": suite.get("version"),
        "run_id": run_id,
        "started_at": now,
        "completed_at": now,
        "runner": {"name": runner_name, "version": runner_version},
        "results": results,
        "metadata": {"openeval": {"source": "crewai"}},
    }
