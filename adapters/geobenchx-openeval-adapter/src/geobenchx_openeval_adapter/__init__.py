"""GeoBenchX <-> EvalPort adapter.

Standalone converter between GeoBenchX's Task/Solution/ScoreValues objects
and the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Follows the same playbook as the other adapters in this repo (see e.g.
adapters/crewai-openeval-adapter): it works against GeoBenchX's public
Task/Solution/Step/TaskSet shapes (real pydantic objects, or an equivalent
dict) from the outside via attribute-or-key access, so no change to
GeoBenchX itself is required and no hard runtime dependency on the
`geobenchx` package is needed either -- GeoBenchX isn't published to PyPI
and has no installable package metadata at its repo root anyway (see the
`geobenchx` extra in pyproject.toml for how to get the real classes
alongside this adapter, e.g. for building fixtures).

Discussed and approved in https://github.com/Solirinai/GeoBenchX/issues/3.
Ground truth for the field names/shapes below is `geobenchx/dataclasses.py`,
`geobenchx/constants.py`, and `geobenchx/evaluation.py` as of GeoBenchX
commit 0edf610 (2026-08-31):

- Task: task_ID, task_text, task_labels (List[TaskLabels]),
  reference_solution_description, reference_solutions (List[Solution]),
  generated_solution (Solution), match_score_LLM / match_reasoning_LLM
  (ScoreValues / str), match_score_Human / match_reasoning_Human.
- Solution: steps (List[Step]).
- Step: function_name (str), arguments (dict), comment (Optional[str]).
- ScoreValues (IntEnum): NO_MATCH=0, PARTIAL_MATCH=1, MATCH=2.
- TaskLabels (str Enum): free-text task category labels.
- A reference solution consisting of a single `reject_task` step means the
  task is a deliberately-unsolvable "should the agent decline?" probe --
  this becomes a first-class `expected_tools == ["reject_task"]` /
  `metadata.unsolvable` marker below, instead of a special case downstream
  consumers have to know to look for.
- `EVALUATION_TAXONOMY` below is copied verbatim from
  `geobenchx/evaluation.py` so the exported suite's `llm_judge` grader
  carries the real 0/1/2 scoring rubric GeoBenchX's own judge panel uses,
  not a paraphrase of it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "results_to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"

JUDGE_ID = "geobenchx_llm_judge"

# Copied verbatim from geobenchx/evaluation.py's EVALUATION_TAXONOMY constant
# (GeoBenchX commit 0edf610), so a reader of the exported suite knows exactly
# what the 0/1/2 score means without needing the GeoBenchX source repo.
EVALUATION_TAXONOMY = """
<TAXONOMY>
While evaluating how close the candidate solution matches to the reference solution, you are using matching score.
Matching score = 0 - Candidate solution does not match any of the reference solutions provided for this task, there are essential discrepancies like different non-similar data are used, inaproppriate tools are used or different results are produced
Matching score = 1 - Candidate solution partially matches at least one of the reference solutions provided for this task, there are non-critical discrepancies like color scheme used for mapping.
Matching score = 2 - Candidate solution fully matches one of the reference solutions provided for this task, there are only non-essential discrepancies like wording of the map's legend.
</TAXONOMY>
"""

# The GeoBenchX citation the maintainer (Varvara, Solirinai/GeoBenchX#3) asked
# to be included in the exported suite's metadata. Matches CITATION.cff's
# preferred-citation exactly (DOI 10.1145/3764915.3770721).
GEOBENCHX_CITATION: Dict[str, Any] = {
    "text": (
        "Krechetova, Varvara; Kochedykov, Denis. \"GeoBenchX: Benchmarking LLMs "
        "in Agent Solving Multistep Geospatial Tasks.\" Proceedings of the 1st "
        "ACM SIGSPATIAL International Workshop on Generative and Agentic AI for "
        "Multi-Modality Space-Time Intelligence (GeoGenAgent '25), ACM, 2025, "
        "pp. 27-35."
    ),
    "doi": "10.1145/3764915.3770721",
    "url": "https://doi.org/10.1145/3764915.3770721",
    "software_url": "https://github.com/Solirinai/GeoBenchX",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    GeoBenchX's Task/Solution/Step are pydantic BaseModels, but plain dicts
    with the same field names (e.g. round-tripped through JSON) show up too,
    so every accessor in this module goes through here rather than assuming
    one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _enum_value(x: Any) -> Any:
    """Return the plain value of an Enum/IntEnum, or pass through a plain value."""
    return getattr(x, "value", x)


def _step_dict(step: Any) -> Dict[str, Any]:
    """Normalize a GeoBenchX Step (object or dict) to a plain, JSON-safe dict."""
    return {
        "function_name": _get(step, "function_name"),
        "arguments": dict(_get(step, "arguments") or {}),
        "comment": _get(step, "comment"),
    }


def _solution_steps(solution: Any) -> List[Dict[str, Any]]:
    if solution is None:
        return []
    return [_step_dict(s) for s in (_get(solution, "steps") or [])]


def _solution_repr(solution: Any) -> str:
    """Readable `func(arg=val, ...)  # comment` text for a Solution, one step per line."""
    lines = []
    for step in _solution_steps(solution):
        args_str = ", ".join(f"{k}={v!r}" for k, v in step["arguments"].items())
        line = f"{step['function_name']}({args_str})"
        if step["comment"]:
            line += f"  # {step['comment']}"
        lines.append(line)
    return "\n".join(lines)


def _reference_solutions_payload(task: Any) -> List[List[Dict[str, Any]]]:
    return [_solution_steps(sol) for sol in (_get(task, "reference_solutions") or [])]


def _expected_tools(task: Any) -> List[str]:
    """Union of function_names across every reference solution, sorted.

    Flattening across `reference_solutions` (rather than picking just one)
    is deliberate: it's what makes a `reject_task`-only reference solution a
    first-class, checkable `expected_tools == ["reject_task"]` case instead
    of a special-cased empty/absent value downstream.
    """
    names = set()
    for sol in _get(task, "reference_solutions") or []:
        for step in _get(sol, "steps") or []:
            fn = _get(step, "function_name")
            if fn:
                names.add(fn)
    return sorted(names)


def _task_labels(task: Any) -> List[str]:
    return [_enum_value(l) for l in (_get(task, "task_labels") or [])]


def _judge_grader() -> Dict[str, Any]:
    return {
        "id": JUDGE_ID,
        "type": "llm_judge",
        "description": (
            "0/1/2 match score from GeoBenchX's LLM-judge panel "
            "(geobenchx.evaluation.score_task_solution), comparing the candidate "
            "solution's tool-call sequence against one or more manually-authored "
            "reference solutions listed in this test case's "
            "metadata.reference_solutions."
        ),
        "params": {
            # GeoBenchX's judge model is caller-selectable (see
            # geobenchx.constants for the supported MODEL_* identifiers and
            # evaluation.score_task_solution's `model` parameter) rather than
            # one fixed model, so this documents where to look instead of
            # asserting a single model id that would be wrong for most runs.
            "model": "configurable; see geobenchx.evaluation.score_task_solution(model=...)",
            "prompt": (
                "Task: {input}\n\nCandidate solution (tool calls): {output}\n"
                + EVALUATION_TAXONOMY
                + "\nScore the candidate solution's match to the task's reference "
                "solution(s) (see this test case's metadata.reference_solutions) "
                "using the taxonomy above. Return 0, 1, or 2."
            ),
        },
    }


def _task_to_testcase(task: Any) -> Dict[str, Any]:
    task_id = _get(task, "task_ID")
    if not task_id:
        raise ValueError("GeoBenchX task is missing task_ID")

    expected_tools = _expected_tools(task)
    tc: Dict[str, Any] = {
        "id": str(task_id),
        "input": _get(task, "task_text") or "",
        "graders": [JUDGE_ID],
        "tags": _task_labels(task),
        "metadata": {
            "reference_solutions": _reference_solutions_payload(task),
            "reference_solution_description": _get(task, "reference_solution_description"),
            "unsolvable": expected_tools == ["reject_task"],
        },
    }
    if expected_tools:
        tc["expected_tools"] = expected_tools
    return tc


def to_openeval(task_set: Any, suite_id: str = "geobenchx") -> Dict[str, Any]:
    """Export a GeoBenchX TaskSet (or any iterable of GeoBenchX-shaped Tasks)
    to an EvalPort-shaped EvalSuite (dict).

    `task_set` may be a real `geobenchx.dataclasses.TaskSet` (iterable over
    its `.tasks`), a plain list of `Task` objects, or a list of dicts with
    the same field names -- no direct `geobenchx` import is required.

    Each `Task` becomes one `TestCase`:
    - `id = task.task_ID`, `input = task.task_text`,
      `tags = [l.value for l in task.task_labels]`.
    - `expected_tools` = the union of `step.function_name` across every
      `reference_solutions[*].steps`, sorted. A `reject_task`-only reference
      solution therefore surfaces as `expected_tools == ["reject_task"]`
      (also mirrored in `metadata.unsolvable`) instead of an empty/absent
      field a consumer has to special-case.
    - `metadata.reference_solutions` carries every reference solution's
      steps (function_name/arguments/comment) verbatim, so nothing about
      *how* to solve the task is lost by reducing it to `expected_tools`.

    Every `TestCase` references the single `gr_geobenchx_llm_judge` grader,
    whose `params.prompt` embeds GeoBenchX's actual 0/1/2 `EVALUATION_TAXONOMY`
    text verbatim.

    The suite's `metadata.citation` carries the GeoBenchX citation
    (DOI 10.1145/3764915.3770721) per Solirinai/GeoBenchX#3.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass it
    to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    tasks = list(task_set)
    test_cases = [_task_to_testcase(t) for t in tasks]

    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "name": "GeoBenchX",
        "description": (
            "GeoBenchX: benchmarking LLM agents on multistep geospatial tasks "
            "against manually-authored, tool-call-level reference solutions."
        ),
        "test_cases": test_cases,
        "graders": [_judge_grader()],
        "metadata": {
            "citation": GEOBENCHX_CITATION,
            "openeval": {"source": "geobenchx"},
        },
    }

    ts_metadata = _get(task_set, "metadata", None)
    if ts_metadata:
        suite["metadata"]["geobenchx_metadata"] = dict(ts_metadata)

    return suite


def results_to_openeval(
    task_set: Any,
    run_id: str,
    started_at: str,
    suite_id: str = "geobenchx",
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Export scored GeoBenchX tasks (i.e. `task.match_score_LLM is not None`,
    the state after `geobenchx.evaluation.score_solutions_set()` has run) to
    an EvalPort-shaped ResultSet (dict).

    One `Result` per scored task:
    - `test_case_id = task.task_ID`, `actual_output` = a readable
      `func(args)  # comment` rendering of `task.generated_solution`.
    - One `GraderResult` with `grader_id="gr_geobenchx_llm_judge"`,
      `score = match_score_LLM.value / 2.0` (normalized to EvalPort's
      required [0.0, 1.0] range per SPEC.md Validation Rule 5),
      `passed = (match_score_LLM == ScoreValues.MATCH)`, and
      `reason = match_reasoning_LLM`.
    - The original 0/1/2 score is preserved, un-lossily, under
      `GraderResult.metadata.raw_score_0_1_2` -- and, when present,
      `match_score_Human`/`match_reasoning_Human` under
      `metadata.human_score`/`metadata.human_reasoning` -- so nothing is
      thrown away by the 0-1 normalization SPEC.md Rule 5 requires.

    Tasks with `match_score_LLM is None` (not yet evaluated) are skipped --
    call this only after `score_solutions_set()`/`score_task_solution()` has
    run, the same precondition GeoBenchX's own
    `evaluation.generate_eval_stats()` assumes.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    results: List[Dict[str, Any]] = []
    for task in task_set:
        score = _get(task, "match_score_LLM")
        if score is None:
            continue
        score_int = int(_enum_value(score))

        grader_metadata: Dict[str, Any] = {"raw_score_0_1_2": score_int}
        human_score = _get(task, "match_score_Human")
        if human_score is not None:
            grader_metadata["human_score"] = int(_enum_value(human_score))
            human_reasoning = _get(task, "match_reasoning_Human")
            if human_reasoning:
                grader_metadata["human_reasoning"] = human_reasoning

        grader_result = {
            "grader_id": JUDGE_ID,
            "type": "llm_judge",
            "score": score_int / 2.0,
            "passed": score_int == 2,  # ScoreValues.MATCH
            "reason": _get(task, "match_reasoning_LLM"),
            "metadata": grader_metadata,
        }

        results.append(
            {
                "test_case_id": str(_get(task, "task_ID")),
                "actual_output": _solution_repr(_get(task, "generated_solution")),
                "grader_results": [grader_result],
                "passed": grader_result["passed"],
                "metadata": {},
            }
        )

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "metadata": {"citation": GEOBENCHX_CITATION},
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    return result_set


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of GeoBenchX-shaped Task dicts.

    Returns plain dicts with GeoBenchX `Task` field names
    (task_ID/task_text/task_labels/reference_solution_description/
    reference_solutions), reconstructed from each `TestCase`'s `id`/`input`/
    `tags`/`metadata.reference_solutions`/`metadata.reference_solution_description`
    -- i.e. the inverse of the `TestCase` shape `to_openeval()` produces, so a
    suite this adapter exported round-trips back through `from_openeval()`.

    Each dict's `reference_solutions` entries are `{"steps": [...]}` dicts
    (each step a `function_name`/`arguments`/`comment` dict), so they pass
    straight into `geobenchx.dataclasses.Solution(**sol)` /
    `Step(**step)`, and the whole dict into `Task(**task_dict)` -- no direct
    `geobenchx` import is required by this function itself.

    A `TestCase` with no `metadata.reference_solutions` (e.g. a suite
    authored outside this adapter) yields `reference_solutions: []`, which
    is also a valid, if information-poor, GeoBenchX `Task`.
    """
    tasks: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        metadata = tc.get("metadata") or {}
        reference_solutions = [
            {"steps": steps} for steps in (metadata.get("reference_solutions") or [])
        ]
        tasks.append(
            {
                "task_ID": tc.get("id"),
                "task_text": tc.get("input"),
                "task_labels": tc.get("tags") or [],
                "reference_solution_description": metadata.get("reference_solution_description"),
                "reference_solutions": reference_solutions,
            }
        )
    return tasks
