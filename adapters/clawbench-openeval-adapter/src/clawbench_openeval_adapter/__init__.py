"""ClawBench <-> EvalPort adapter.

Standalone converter between ClawBench's own result output
(``run-meta.json``, the per-run judge verdict file, and the per-batch
``rescore-summary.json``) and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside
ClawBench itself: per TIGER-AI-Lab/ClawBench#322, maintainer Perry2004
confirmed this is a results-side conversion, not a benchmark adapter in
ClawBench's own sense (``clawbench-harbor-adapt`` /
``clawbench-edgebench-adapt`` convert *test case* definitions the other
direction), and suggested it land as a simple script rather than a new
package under ``adapters/`` in ClawBench's own repo. This package is built
and tested against ClawBench's real, current source first; where it ships
(a ``script/`` file in ClawBench directly, vs. here in EvalPort's own
``adapters/``) depends on write access at the time it's opened -- see that
issue thread for the up-to-date status. Same situation, same resolution,
as this repo's ``terminal-bench-science-openeval-adapter``.

What ClawBench actually emits, verified against the real repository (not
just ``docs/scoring.md``, which describes an idealized/older shape):

* ``run-meta.json`` (one per run) is built by ``make_run_meta()`` in
  ``src/clawbench/runner/run_support/metadata.py``. It carries
  ``test_case``, ``instruction``, ``model``, ``harness``, ``intercepted``,
  ``result_category``, ``failure_category``, ``adjusted_eligible``, among
  other fields. It does **not** carry ``judge_match`` or ``final_pass`` --
  those are documented in ``docs/scoring.md`` as merged in, but the actual
  merge point in current code is the rescoring step below, and even there
  no field is literally named ``final_pass``.
* A per-run judge verdict lives in its own file (``judge_llm.json`` for the
  default "lenient" rubric, ``judge.json`` for "strict" -- see
  ``JUDGE_FILE`` in ``src/clawbench/eval/rescore.py``), with a ``match``
  key (``True``/``False``/``None``) and a ``reason`` string.
* ``rescore-summary.json`` (one per batch, written by
  ``aggregate_batch()`` in ``src/clawbench/eval/rescore.py``) rolls a
  batch of runs into ``n_total``, ``n_intercepted``, and a ``tasks`` list
  of per-task rows shaped ``{"task_id", "test_case", "intercepted",
  "match_<rubric>", "reason_<rubric>"}`` for each rubric that was run
  (``rubrics`` lists which). It also carries legacy alias keys
  (``n_judge_match``, ``pass_rate_with_judge``, ...) when the "strict" or
  "lenient" rubric ran. Critically, ``tasks[]`` rows do **not** carry
  ``instruction``/``model``/``harness`` -- those live only in each run's
  own ``run-meta.json``. This adapter merges the two when a caller
  supplies both; it does not fabricate a merge that ClawBench's own code
  doesn't perform.

Mapping to EvalPort's data model
(https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md):

* One ClawBench run -> one EvalPort ``Result``. ``test_case`` (or
  ``task_id`` as fallback) becomes ``test_case_id``. The two-stage
  scoring pipeline (interception, then LLM judge) becomes two
  ``GraderResult`` entries -- ``gr_interception`` and ``gr_judge_match`` --
  so the mechanism ClawBench actually uses is visible in the result, not
  collapsed into one opaque score. ``Result.passed`` is
  ``intercepted AND judge_match is True``, matching the
  ``final_pass = intercepted AND judge_match`` rule documented in
  ``docs/scoring.md``.
* A batch's ``rescore-summary.json`` -> one EvalPort ``ResultSet``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "run_to_result",
    "to_openeval",
    "from_openeval",
    "INTERCEPTION_GRADER_ID",
    "JUDGE_GRADER_ID",
    "__version__",
]
__version__ = "0.1.0"

INTERCEPTION_GRADER_ID = "gr_interception"
JUDGE_GRADER_ID = "gr_judge_match"


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def run_to_result(
    run_meta: Dict[str, Any],
    judge: Optional[Dict[str, Any]] = None,
    rubric: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one ClawBench run into an EvalPort ``Result`` dict.

    ``run_meta`` is the dict already parsed from that run's
    ``run-meta.json`` (or, when only a ``rescore-summary.json`` task row
    is available, a minimal stand-in with at least ``test_case``/
    ``task_id`` and ``intercepted`` -- see ``to_openeval()``).

    ``judge`` is the dict already parsed from that run's judge verdict
    file (``judge_llm.json``/``judge.json``, or the ``match``/``reason``
    pulled from a ``rescore-summary.json`` task row for a given rubric).
    Pass ``None`` when the run was never judged (e.g. ``intercepted`` was
    already ``False``, so Stage 2 never ran per ``docs/scoring.md``).

    Two ``GraderResult`` entries are emitted, mirroring ClawBench's real
    two-stage pipeline:

    - ``gr_interception``: score/passed from ``intercepted`` alone.
    - ``gr_judge_match``: only present when ``judge`` is given. score is
      ``1.0``/``0.0``/``None`` for ``match`` True/False/None (a judge
      that "could not decide" -- rare, per ``docs/scoring.md`` -- carries
      a null score and counts as not-passed, exactly like ClawBench's own
      aggregate treats it).

    ``Result.passed`` is ``intercepted AND match is True``, i.e. the
    ``final_pass`` rule from ``docs/scoring.md``.
    """
    test_case_id = _get(run_meta, "test_case") or _get(run_meta, "task_id")
    if not test_case_id:
        raise ValueError("run_meta must have a 'test_case' or 'task_id' to become a Result.test_case_id")

    intercepted = bool(_get(run_meta, "intercepted"))

    grader_results: List[Dict[str, Any]] = [
        {
            "grader_id": INTERCEPTION_GRADER_ID,
            "type": "custom",
            "score": 1.0 if intercepted else 0.0,
            "passed": intercepted,
            "reason": "final request matched eval_schema" if intercepted else "final request did not match eval_schema (or agent never reached it)",
            "metadata": {"handler": "clawbench:interception"},
        }
    ]

    judge_match: Optional[bool] = None
    if judge is not None:
        judge_match = _get(judge, "match")
        score = 1.0 if judge_match is True else (0.0 if judge_match is False else None)
        gr: Dict[str, Any] = {
            "grader_id": JUDGE_GRADER_ID,
            "type": "llm_judge",
            "score": score,
            "passed": judge_match is True,
            "metadata": {"handler": "clawbench:llm_judge"},
        }
        reason = _get(judge, "reason")
        if reason:
            gr["reason"] = reason
        judge_model = _get(judge, "judge_model")
        if judge_model:
            gr["metadata"]["judge_model"] = judge_model
        grader_results.append(gr)

    passed = bool(intercepted and judge_match is True)

    metadata: Dict[str, Any] = {}
    for key in ("result_category", "failure_category", "adjusted_eligible", "model", "harness", "task_id"):
        value = _get(run_meta, key)
        if value is not None:
            metadata[key] = value
    if rubric:
        metadata["rubric"] = rubric

    result: Dict[str, Any] = {
        "test_case_id": str(test_case_id),
        "passed": passed,
        "grader_results": grader_results,
    }
    # ClawBench scores an intercepted HTTP request, not a text completion, so
    # there is no honest value for Result.actual_output here; the instruction
    # that was scored against goes under metadata instead.
    instruction = _get(run_meta, "instruction")
    if instruction is not None:
        metadata["instruction"] = instruction
    duration = _get(run_meta, "duration_seconds")
    if isinstance(duration, (int, float)):
        result["duration_ms"] = int(round(duration * 1000))
    if metadata:
        result["metadata"] = metadata
    return result


def to_openeval(
    rescore_summary: Dict[str, Any],
    run_metas: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
    rubric: Optional[str] = None,
    suite_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a ClawBench batch's ``rescore-summary.json`` to an EvalPort ``ResultSet`` dict.

    ``rescore_summary`` is the dict already parsed from that batch's
    ``rescore-summary.json`` (real shape: see module docstring / the
    ``aggregate_batch()`` source in ``src/clawbench/eval/rescore.py``).

    ``run_metas``, if given, maps each run's ``test_case`` value to that
    run's already-parsed ``run-meta.json`` dict, so ``instruction``,
    ``model``, and ``harness`` can be carried into each ``Result`` --
    ``rescore-summary.json``'s own ``tasks[]`` rows don't carry those
    fields. Omit it (or leave a given ``test_case`` unmapped) and the
    corresponding ``Result`` is still produced, just without that
    enrichment -- nothing is fabricated to fill the gap.

    ``run_id`` and ``started_at`` are required: ClawBench's
    ``rescore-summary.json`` records neither a run id nor a start
    timestamp for the batch, so there is nothing honest to default them
    to. Pass your own (e.g. the batch directory name, and the mtime of
    its earliest ``run-meta.json``).

    ``rubric`` selects which of ``rescore_summary["rubrics"]``
    (``"lenient"`` and/or ``"strict"``) to score against; defaults to the
    first entry in ``rescore_summary["rubrics"]``, or ``"lenient"`` if
    that key is absent (matching ``clawbench-rescore``'s own default).
    """
    run_metas = run_metas or {}
    rubrics = rescore_summary.get("rubrics") or ["lenient"]
    active_rubric = rubric or rubrics[0]

    results: List[Dict[str, Any]] = []
    for task_row in rescore_summary.get("tasks", []):
        test_case = task_row.get("test_case")
        base_run_meta = run_metas.get(test_case) if test_case is not None else None
        if base_run_meta is None:
            base_run_meta = {
                "test_case": test_case,
                "task_id": task_row.get("task_id"),
                "intercepted": task_row.get("intercepted"),
            }
        match_key = f"match_{active_rubric}"
        reason_key = f"reason_{active_rubric}"
        judge = None
        # rescore.py's aggregate_batch() writes match_<rubric>/reason_<rubric> for
        # every task row unconditionally (defaulting to match=None, reason="" when
        # there's no judge file) -- but rescore_one() only ever judges a run when
        # it was intercepted; Stage 2 never runs otherwise. So the presence of the
        # key alone doesn't mean a judge actually ran -- gate on `intercepted` too,
        # matching that real control flow, rather than emitting a spurious
        # gr_judge_match grader for a run that was never judged.
        if task_row.get("intercepted") and match_key in task_row:
            judge = {
                "match": task_row.get(match_key),
                "reason": task_row.get(reason_key),
                "judge_model": rescore_summary.get("judge_model"),
            }
        results.append(run_to_result(base_run_meta, judge, rubric=active_rubric))

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    n_intercepted = sum(1 for r in results if r["grader_results"][0]["passed"])

    batch_dir = rescore_summary.get("batch_dir")
    resolved_suite_id = suite_id or (
        f"clawbench_{batch_dir.rstrip('/').rsplit('/', 1)[-1]}" if batch_dir else "clawbench_batch"
    )

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": resolved_suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "runner": {"name": "clawbench", "version": "n/a"},
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total) if total else 0.0,
        },
        "metadata": {
            "openeval": {"source": "clawbench"},
            "clawbench_batch_dir": batch_dir,
            "clawbench_rubric": active_rubric,
            "clawbench_n_intercepted": n_intercepted,
        },
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    return result_set


def from_openeval(result_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort ``ResultSet`` into ClawBench-``rescore-summary``-shaped rows.

    Intentionally best-effort/lossy, the same way this repo's other
    adapters are for their reverse direction: EvalPort's ``Result`` has no
    first-class slot for ClawBench's ``result_category``/
    ``failure_category``/``adjusted_eligible`` taxonomy, so those are only
    recovered when they were carried through under ``Result.metadata`` by
    this adapter's own ``to_openeval()`` (or a producer following the same
    convention) -- round-tripping a ``ResultSet`` this adapter did not
    produce carries through whatever it finds under ``metadata``,
    verbatim, and nothing more.
    """
    rows: List[Dict[str, Any]] = []
    for result in result_set.get("results", []):
        metadata = result.get("metadata") or {}
        grader_by_id = {gr.get("grader_id"): gr for gr in result.get("grader_results", [])}
        interception = grader_by_id.get(INTERCEPTION_GRADER_ID, {})
        judge = grader_by_id.get(JUDGE_GRADER_ID)
        row: Dict[str, Any] = {
            "test_case": result.get("test_case_id"),
            "task_id": metadata.get("task_id"),
            "intercepted": interception.get("passed", False),
        }
        if judge is not None:
            match_score = judge.get("score")
            row["match"] = None if match_score is None else bool(judge.get("passed"))
            row["reason"] = judge.get("reason")
        for key in ("result_category", "failure_category", "adjusted_eligible", "model", "harness", "instruction"):
            if key in metadata:
                row[key] = metadata[key]
        rows.append(row)
    return rows
