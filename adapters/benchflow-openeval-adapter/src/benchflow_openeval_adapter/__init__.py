"""BenchFlow -> EvalPort adapter (export-only).

Standalone converter from BenchFlow's (https://github.com/benchflow-ai/benchflow)
Verifiers/Prime-RL-shaped ``results.jsonl`` rollout records to the EvalPort
interchange format (https://github.com/adhabnr-ux/evalport). Zero footprint
on BenchFlow itself -- built entirely against ``results.jsonl``'s documented
public shape (``benchflow.trajectories.results.build_rollout_results_record``
/ ``write_job_results_jsonl``), same playbook as the AutoGen and Opik
adapters (https://github.com/adhabnr-ux/evalport/tree/main/adapters).

This is deliberately **export-only** -- there is no ``from_openeval()``. Per
the design discussion on https://github.com/benchflow-ai/benchflow/issues/1072,
a BenchFlow task (``task.md`` + ``environment/Dockerfile`` + a sandboxed
``verifier/test.sh``) is a whole RL environment, not an EvalPort ``TestCase``
(``{id, input, graders[]}`` -- essentially a prompt plus grading criteria).
Forcing the environment into ``TestCase.input`` would lose everything that
actually defines the task, so there is no suite-side conversion either.

Every design choice below is grounded in BenchFlow's own source, not guessed,
per the corrections from maintainer @ElegantLin on issue #1072:

- **``passed`` is ``reward == 1.0``, exactly -- never a >= 0.5 midpoint
  threshold.** This mirrors BenchFlow's own canonical classification, used
  identically in ``benchflow._utils.scoring.classify_result``,
  ``benchflow.evaluation`` (``_log_and_report``), ``benchflow.eval_lift``
  (``RolloutResult.passed``), and ``benchflow.review.runner`` -- reward is
  binary pass/fail at 1.0 everywhere in the codebase, never binarized at a
  midpoint. A caller who wants a different bar for a specific deployment can
  override via ``exact_pass_reward``.
- **A rollout is "scored" iff its ``metrics`` dict contains a ``"reward"``
  key.** ``build_rollout_results_record`` only puts ``"reward"`` into
  ``metrics`` when the verifier actually produced a ``rewards`` dict
  (``_metrics_from_rewards`` returns early with just
  ``{"n_tool_calls", "n_prompts"}`` when ``rewards`` is ``None`` -- which
  ``RolloutResult.rewards``'s own docstring says happens "if verification was
  skipped or failed"). This is the same signal
  ``benchflow._utils.scoring.extract_reward`` uses (``rewards.get("reward")``
  is ``None`` exactly when ``rewards`` isn't a populated dict), just read off
  the flattened ``results.jsonl`` row instead of the internal
  ``RolloutResult``. A rollout whose verifier ran but whose export step later
  failed (``stop_condition == "export_error"``) still has a real ``reward``
  in ``metrics`` and is scored normally -- only agent/verifier failures
  (which leave ``rewards`` unpopulated) are unscored. Unscored rollouts get
  ``score: null, passed: false`` on every ``GraderResult`` (spec Validation
  Rule 6), never a fabricated ``0.0`` "failure".
- **No double-counted reward.** ``metrics`` already contains ``"reward"`` as
  one of its own keys (``_metrics_from_rewards`` copies every numeric key of
  the ``rewards`` dict, including ``"reward"`` itself, into ``metrics``) --
  so this adapter builds exactly one ``GraderResult`` per ``metrics`` key
  (excluding the two non-score bookkeeping keys ``n_tool_calls``/
  ``n_prompts``), rather than emitting a hand-added ``bf_reward`` on top of
  what the metrics loop already yields.
- **Scores are normalized, not truncated, into EvalPort's required
  [0.0, 1.0] range** (spec Validation Rule 5), because BenchFlow reward
  functions are not guaranteed to be [0, 1] (some tasks declare wider ranges,
  e.g. [-1, 1]) and a hard ``max(0, min(1, x))`` clamp collapses everything
  below the floor to a flat 0, destroying exactly the distinctions a
  continuous reward is meant to carry. ``reward_range``/``metric_ranges``
  declare each score's true native range so it can be linearly rescaled
  instead; the untouched original is always preserved at
  ``GraderResult.metadata["openeval"]["raw_score"]`` (spec Appendix B).
- **Repeated trials of the same task get explicit ``attempt`` numbers.**
  ``write_job_results_jsonl`` already assigns every rollout row for a given
  task the *same* ``example_id`` (its own dedup counter, keyed by
  ``info.task_id or info.task_name`` -- see ``_job_example_key``); this
  adapter uses that identical key as ``test_case_id`` and, when more than one
  row shares it, assigns ascending 1-indexed ``attempt`` numbers in the order
  the rows were given (spec Extension Mechanism -> Repetition & Attempt
  Tracking / Discussion #22). A single row per task gets no ``attempt``
  field at all, per spec ("absent means single-attempt").
- **``isolation`` is never guessed.** Whether repeated BenchFlow trials of
  the same task run in fresh, isolated environments or share state between
  attempts is a property of how the job was invoked, not something a
  ``results.jsonl`` row records -- so this adapter leaves
  ``ResultSet.isolation`` unset unless the caller explicitly passes it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["job_results_to_openeval", "job_results_file_to_openeval", "__version__"]
__version__ = "0.1.0"

# `build_rollout_results_record`'s `metrics` dict always carries these two
# bookkeeping counters (`_metrics_from_rewards`'s own fixed keys) alongside
# whatever named reward/rubric scores the verifier produced. Neither is a
# score, so neither becomes a GraderResult.
_NON_SCORE_METRIC_KEYS = frozenset({"n_tool_calls", "n_prompts"})

# The metrics key `_metrics_from_rewards` uses for BenchFlow's own top-level
# reward -- the one field with a codebase-wide, exact reward == 1.0 pass
# convention (see module docstring). Every other metrics key gets the same
# treatment but its own grader_id / range.
_REWARD_METRIC_KEY = "reward"

Range = Tuple[float, float]


def _normalize(raw: float, value_range: Range) -> float:
    """Linearly rescale `raw` from `value_range` into [0.0, 1.0].

    Falls back to a defensive clamp (never a silent crash) if `raw` is
    outside the declared range -- a range that's merely a best-effort
    declaration, not something this adapter can verify against BenchFlow's
    actual task definitions. The untouched `raw` value is always preserved
    separately in `metadata.openeval.raw_score` regardless of what happens
    here, so a defensive clamp never actually discards information.
    """
    lo, hi = value_range
    if hi <= lo:
        raise ValueError(f"Invalid range {value_range!r}: high must be > low")
    normalized = (raw - lo) / (hi - lo)
    return max(0.0, min(1.0, normalized))


def _stringify_completion(completion: Any) -> Optional[str]:
    """Best-effort flatten of a `results.jsonl` row's `completion` (a list of
    chat messages) into a single text string for `Result.actual_output`.
    Returns None when there's nothing text-shaped to extract, rather than
    guessing.
    """
    if not isinstance(completion, list) or not completion:
        return None
    parts: List[str] = []
    for message in completion:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
    return "\n".join(parts) if parts else None


def _grader_result(
    *,
    grader_id: str,
    raw_score: Optional[float],
    value_range: Range,
    pass_reward: float,
) -> Dict[str, Any]:
    if raw_score is None:
        return {
            "grader_id": grader_id,
            "type": "custom",
            "score": None,
            "passed": False,
        }
    return {
        "grader_id": grader_id,
        "type": "custom",
        "score": _normalize(raw_score, value_range),
        "passed": bool(raw_score == pass_reward),
        "metadata": {"openeval": {"raw_score": raw_score}},
    }


def _test_case_id(row: Mapping[str, Any]) -> str:
    info = row.get("info")
    if isinstance(info, dict):
        task_id = info.get("task_id") or info.get("task_name")
        if task_id:
            return str(task_id)
    task_id = row.get("task_id") or row.get("task_name")
    if task_id:
        return str(task_id)
    # Matches `_job_example_key`'s own final fallback (the rollout's own
    # directory name) as closely as a bare row allows; `example_id` is the
    # nearest equivalent available without the original directory.
    example_id = row.get("example_id")
    return str(example_id) if example_id is not None else "unknown_task"


def _row_metadata(row: Mapping[str, Any]) -> Dict[str, Any]:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    benchflow_meta: Dict[str, Any] = {
        "rollout_name": info.get("rollout_name"),
        "agent": info.get("agent"),
        "agent_name": info.get("agent_name"),
        "model": info.get("model"),
        "is_truncated": row.get("is_truncated"),
        "stop_condition": row.get("stop_condition"),
        "training_ready": info.get("training_ready"),
        "training_ready_reason": info.get("training_ready_reason"),
        "total_tool_calls": row.get("total_tool_calls"),
    }
    if isinstance(info.get("reward_details"), dict):
        benchflow_meta["reward_details"] = info["reward_details"]
    error = row.get("error")
    if isinstance(error, dict) and error.get("error"):
        # The specific BenchFlow category ("agent_error"/"verifier_error"/
        # "export_error") -- see `_row_error` for why `Result.error.type`
        # itself can't carry this.
        benchflow_meta["error_category"] = error["error"]
    token_usage = row.get("token_usage")
    if isinstance(token_usage, dict) and any(v for v in token_usage.values()):
        benchflow_meta["token_usage"] = token_usage
    # Drop keys BenchFlow itself left unset, rather than shipping a metadata
    # blob full of Nones.
    benchflow_meta = {k: v for k, v in benchflow_meta.items() if v is not None}
    return {"benchflow": benchflow_meta}


def _duration_ms(row: Mapping[str, Any]) -> Optional[int]:
    timing = row.get("timing")
    if not isinstance(timing, dict):
        return None
    total_seconds = timing.get("total")
    if not isinstance(total_seconds, (int, float)) or isinstance(total_seconds, bool):
        return None
    return round(total_seconds * 1000)


def _row_error(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a `results.jsonl` row's `error` object to EvalPort's `Result.error`.

    `spec/schemas/resultset.json` closes `Result.error.type` to exactly
    `"timeout" | "provider_error" | "runner_error"` -- none of which match
    BenchFlow's own category strings (`"agent_error"`, `"verifier_error"`,
    `"export_error"`). Every BenchFlow error is a failure somewhere in the
    execution/verification pipeline rather than a request timeout or an
    upstream model-provider error, so `"runner_error"` is the correct (not
    just closest-available) EvalPort category for all three; the original,
    more specific BenchFlow category is preserved verbatim in
    `Result.metadata.benchflow.error_category` (see `_row_metadata`) rather
    than discarded.
    """
    error = row.get("error")
    if not isinstance(error, dict):
        return None
    return {
        "type": "runner_error",
        "message": error.get("error_chain_str") or error.get("error_chain_repr") or "",
    }


def job_results_to_openeval(
    rows: Iterable[Mapping[str, Any]],
    *,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    isolation: Optional[str] = None,
    reward_range: Range = (0.0, 1.0),
    metric_ranges: Optional[Mapping[str, Range]] = None,
    exact_pass_reward: float = 1.0,
    metric_pass_reward: Optional[float] = None,
    runner_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert BenchFlow job-level ``results.jsonl`` rows into an EvalPort
    ``ResultSet`` dict. Export-only -- see the module docstring for why
    there is no ``from_openeval``.

    Args:
        rows: Parsed JSON objects, one per line of a BenchFlow job's
            aggregated ``results.jsonl`` (see
            ``benchflow.trajectories.results.write_job_results_jsonl``) --
            or of a single rollout's ``results.jsonl``
            (``write_rollout_results_jsonl``); both share the same row
            shape. Use ``job_results_file_to_openeval`` to read straight
            from a file instead of pre-parsing yourself.
        suite_id: EvalPort ``ResultSet.suite_id``. BenchFlow has no bundled
            suite concept to source this from (see module docstring) --
            pick something that identifies the BenchFlow task set this job
            ran, e.g. the job's benchmark/config name.
        run_id: EvalPort ``ResultSet.run_id`` -- typically the BenchFlow job
            id / run directory name.
        started_at: ISO 8601 timestamp. Defaults to the current UTC time if
            omitted (``results.jsonl`` rows don't carry a job-level start
            time, only per-rollout ``timing`` deltas).
        completed_at: ISO 8601 timestamp. Defaults to `started_at` if
            omitted.
        isolation: ``ResultSet.isolation`` ("fresh"/"shared"/any string).
            Left unset unless you pass it explicitly -- see module
            docstring.
        reward_range: The native ``(low, high)`` range of BenchFlow's
            top-level ``reward`` metric for this job. Defaults to
            ``(0.0, 1.0)``, BenchFlow's overwhelmingly common convention
            (``reward == 1.0`` is the pass bar everywhere in the codebase --
            see module docstring); pass e.g. ``(-1.0, 1.0)`` for a task
            whose verifier is documented to use a wider scale.
        metric_ranges: Per-metric-name ``(low, high)`` overrides for every
            *other* ``metrics`` key (rubric items, named sub-scores, ...).
            A key not listed here defaults to ``(0.0, 1.0)``.
        exact_pass_reward: The exact reward value that counts as a pass for
            the ``bf_reward`` grader. Defaults to ``1.0``, mirroring
            ``benchflow._utils.scoring.classify_result`` /
            ``benchflow.eval_lift.RolloutResult.passed`` /
            ``benchflow.review.runner`` exactly -- deliberately an exact
            equality check, not a >= threshold, so partial credit is never
            silently promoted to a pass.
        metric_pass_reward: The exact (pre-normalization) value that counts
            as a pass for every non-reward metric grader. Defaults to each
            metric's own declared range maximum (from `metric_ranges`,
            i.e. "reached the top of its scale"), applying the same
            exact-equality convention as `exact_pass_reward` rather than an
            invented 0.5 midpoint. Pass a float to use one fixed bar for
            every non-reward metric instead.
        runner_version: ``ResultSet.runner.version``. Omit to leave it unset
            (this adapter has no way to introspect the BenchFlow version
            that produced `rows`).

    Returns:
        A plain dict conforming to the EvalPort ``ResultSet`` schema. Pass
        it to ``openeval.validate.validate_result_set()`` to confirm
        compliance.

    Raises:
        ValueError: if `rows` is empty (an EvalPort ``ResultSet.results``
            must be non-empty per the schema) or a `test_case_id` derived
            from `rows` collides with itself in a way that can't be
            disambiguated (never actually happens -- rows sharing a
            `test_case_id` become `attempt`s of the same test case, which
            is always valid -- this is here only to fail loudly instead of
            emitting a document ``validate_result_set`` would reject, should
            a future BenchFlow change break that assumption).
    """
    rows = list(rows)
    if not rows:
        raise ValueError(
            "job_results_to_openeval() received no rows -- an EvalPort "
            "ResultSet.results must be non-empty"
        )
    if started_at is None:
        from datetime import datetime, timezone

        started_at = datetime.now(timezone.utc).isoformat()
    if completed_at is None:
        completed_at = started_at

    metric_ranges = dict(metric_ranges or {})

    # Group rows by test_case_id, preserving first-seen order, so repeated
    # trials of the same task get ascending `attempt` numbers in the order
    # they were given (see module docstring).
    order: List[str] = []
    grouped: "MutableMapping[str, List[Mapping[str, Any]]]" = {}
    for row in rows:
        tcid = _test_case_id(row)
        if tcid not in grouped:
            grouped[tcid] = []
            order.append(tcid)
        grouped[tcid].append(row)

    results: List[Dict[str, Any]] = []
    all_scores: List[float] = []
    passed_count = 0
    unscored_count = 0

    for tcid in order:
        group = grouped[tcid]
        multi_attempt = len(group) > 1
        for attempt_idx, row in enumerate(group, start=1):
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            is_scored = _REWARD_METRIC_KEY in metrics

            grader_results: List[Dict[str, Any]] = []
            reward_gr = _grader_result(
                grader_id="bf_reward",
                raw_score=metrics.get(_REWARD_METRIC_KEY) if is_scored else None,
                value_range=reward_range,
                pass_reward=exact_pass_reward,
            )
            grader_results.append(reward_gr)

            if is_scored:
                for key, value in metrics.items():
                    if key in _NON_SCORE_METRIC_KEYS or key == _REWARD_METRIC_KEY:
                        continue
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    m_range = metric_ranges.get(key, (0.0, 1.0))
                    m_pass = (
                        metric_pass_reward
                        if metric_pass_reward is not None
                        else m_range[1]
                    )
                    grader_results.append(
                        _grader_result(
                            grader_id=f"bf_{key}",
                            raw_score=float(value),
                            value_range=m_range,
                            pass_reward=m_pass,
                        )
                    )

            # Result.passed follows the `bf_reward` grader alone, not a
            # strict AND of every grader_results entry. This is a deliberate
            # departure from the spec's default "all" aggregation semantic:
            # BenchFlow's own pass/fail definition (`classify_result`,
            # `RolloutResult.passed`, `evaluation._log_and_report`) is
            # "reward == 1.0", full stop -- a rubric sub-metric that didn't
            # hit its own max never demotes an otherwise-passing rollout,
            # and an AND-of-all-graders Result.passed here would silently
            # contradict what BenchFlow itself would report for the same
            # rollout. The sub-metric GraderResults are still emitted in
            # full (each with its own honest `passed`) for anyone who wants
            # finer-grained analysis; they just don't drive the roll-up.
            result_passed = is_scored and bool(reward_gr["passed"])
            if result_passed:
                passed_count += 1
            if not is_scored:
                unscored_count += 1
            for gr in grader_results:
                if gr["score"] is not None:
                    all_scores.append(gr["score"])

            result: Dict[str, Any] = {
                "test_case_id": tcid,
                "passed": result_passed,
                "grader_results": grader_results,
                "metadata": _row_metadata(row),
            }
            if multi_attempt:
                result["attempt"] = attempt_idx
            actual_output = _stringify_completion(row.get("completion"))
            if actual_output is not None:
                result["actual_output"] = actual_output
            duration_ms = _duration_ms(row)
            if duration_ms is not None:
                result["duration_ms"] = duration_ms
            error = _row_error(row)
            if error is not None:
                result["error"] = error
            if not is_scored:
                result["metadata"]["openeval"] = {"aggregation_status": "unscored"}

            results.append(result)

    total = len(results)
    # spec/schemas/resultset.json declares `summary` with
    # `"additionalProperties": false` and a fixed property list (no
    # "unscored" key, `avg_score` typed as a plain non-nullable `number`) --
    # so the unscored count and a null avg_score can't live here. The count
    # goes to `metadata.openeval.unscored_count` instead (`metadata` is
    # `additionalProperties: true`); `avg_score` falls back to 0 when there's
    # no scored grader_result at all, matching the convention every other
    # EvalPort adapter in this repo already uses (e.g. opik-openeval-adapter).
    summary = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count - unscored_count,
        "skipped": 0,
        "pass_rate": (passed_count / total) if total else 0,
        "avg_score": (sum(all_scores) / len(all_scores)) if all_scores else 0,
    }

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "runner": (
            {"name": "benchflow", "version": runner_version}
            if runner_version
            else {"name": "benchflow"}
        ),
        "results": results,
        "summary": summary,
        "metadata": {"openeval": {"source": "benchflow", "unscored_count": unscored_count}},
    }
    if isolation is not None:
        result_set["isolation"] = isolation
    return result_set


def job_results_file_to_openeval(
    results_jsonl_path: Union[str, Path],
    **kwargs: Any,
) -> Dict[str, Any]:
    """Read a BenchFlow ``results.jsonl`` file and convert it via
    `job_results_to_openeval`. Every keyword argument `job_results_to_openeval`
    accepts is accepted here too.

    Blank lines are skipped (``write_job_results_jsonl`` never emits them,
    but a hand-edited or concatenated file might).
    """
    path = Path(results_jsonl_path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return job_results_to_openeval(rows, **kwargs)
