"""humanbound <-> EvalPort adapter.

Standalone converter between humanbound's
(https://github.com/humanbound/humanbound) adversarial-testing output --
`ExperimentResults` (`stats`, `insights`, `posture`, `exec_t`) and `LogEntry`
(per-conversation verdicts) -- and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Built at the upstream maintainer's explicit invitation: see
https://github.com/humanbound/humanbound/issues/131#issuecomment-5885427531
("We'd prefer the standalone package in the EvalPort repo, as you
suggested... Once it's published, share the link here and we'll be happy to
review it for a mention in our docs.").

Mapping, read directly from humanbound's own public schema surface
(`humanbound.schemas` / `humanbound_cli/engine/schemas.py`, pinned at
humanbound 2.12.0) and its `LocalTestRunner` on-disk result layout
(`humanbound_cli/engine/local_runner.py`, `.humanbound/results/<experiment_id>/`):

| humanbound (`LogEntry`, one per conversation)        | EvalPort `Result` |
|--------------------------------------------------------|----------------------------------------------|
| `thread_id`                                             | `test_case_id`                                |
| `conversation: list[Turn]` (`Turn.u`/`Turn.a`)           | `actual_output` (rendered transcript) + full turns preserved verbatim under `metadata.humanbound.conversation` |
| `result` (`"pass"` \\| `"fail"` \\| `"error"`)             | `passed` (`result == "pass"`); `result == "error"` also sets `Result.error` |
| `exec_t` (seconds -- see `local_runner.py:506`, `f"{exec_t:.1f}s"`) | `duration_ms` (`exec_t * 1000`) |
| `gen_category`, `fail_category`, `severity` (0-100), `confidence` (1-100), `explanation` | one `GraderResult` (`grader_id="humanbound_verdict"`, `type="custom"`): `reason` = `explanation`; `severity`/`confidence`/categories preserved verbatim in `metadata` |

humanbound has no per-conversation "score" field (`severity` grades how bad a
*failure* is, `confidence` grades the judge's certainty in its *own verdict*
-- neither is "how good was this response" on its own). Rather than leave
`GraderResult.score` unset everywhere, or fabricate a number humanbound never
computed, this adapter derives `score = 1 - severity/100` (`severity`
defaults to `0` for a `pass`, so a clean pass scores `1.0`; a `fail`'s score
falls as its severity rises) and is upfront in this module and in
`GraderResult.metadata` that this is **this adapter's own derived score**,
not a humanbound-native metric. `result == "error"` leaves `score` as `None`
(unscored, not zero) since severity is not meaningful for a conversation the
judge never rated. humanbound's own aggregate metrics -- `Stats.reliability`,
`Stats.fail_impact`, `ExperimentPosture` -- are never recomputed by this
adapter; they are carried through verbatim in `ResultSet.metadata.humanbound`
instead, so nothing here contradicts humanbound's own formulas (see
`humanbound_cli/engine/presenter.py`).

`ExperimentResults.stats` / `.posture` / `.insights` / `.exec_t` (the
per-run aggregate) have no per-`Result` home in EvalPort's schema, so they
are carried through verbatim under `ResultSet.metadata.humanbound`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "read_local_results",
    "local_results_to_openeval",
    "__version__",
]
__version__ = "0.1.0"

PathLike = Union[str, "Path"]

# humanbound_cli.engine.schemas.JUDGE_ERROR_CATEGORY -- the fail_category value
# marking a conversation that ran but couldn't be judged (a log whose `result`
# is `"error"` for that reason, not a runner crash). Inlined as a string
# constant, pinned to humanbound 2.12.0, so this module has no import-time
# dependency on the `humanbound` package.
_JUDGE_ERROR_CATEGORY = "judge_error"


def _to_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """Materialize a humanbound pydantic model (or plain dict) into a plain dict.

    Handles pydantic v2 (`model_dump`, called with `by_alias=True` so
    `Stats.pass_` comes back under its real wire key `"pass"`), pydantic v1
    (`.dict()`), plain dicts, and falls back to `vars()` for anything else.
    `model_dump`/`.dict()` recursively serialize nested models too (e.g. a
    `LogEntry.conversation: list[Turn]` comes back as a list of plain
    `{"u":..., "a":...}` dicts), so callers never need to unwrap nested
    fields by hand.

    `Stats`, `ExperimentResults`, and `ExperimentMeta` each override
    `model_dump()` to already force `by_alias=True` internally
    (`humanbound_cli/engine/schemas.py`) and their override's signature
    (`model_dump(self, **kwargs)`) does not accept `by_alias` a second time
    -- passing it raises `TypeError: got multiple values for keyword argument
    'by_alias'`, verified against the real installed classes. Caught here and
    retried with a bare call rather than asking every caller to know which
    humanbound models have this override.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(by_alias=True)
        except TypeError:
            return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(vars(obj))


def _clamp01(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _severity_score(severity: Optional[float]) -> Optional[float]:
    """`1 - severity/100`, clamped to EvalPort's required `[0, 1]`.

    `severity` is documented (and verified against the real judge prompts in
    `humanbound_cli/engine/orchestrators/*/judge.py`) as a 0-100 scale where
    higher is worse. See the module docstring for why this, not `confidence`,
    is what this adapter derives a `GraderResult.score` from.
    """
    if severity is None:
        return None
    return _clamp01(1.0 - (float(severity) / 100.0))


def _render_transcript(conversation: List[Dict[str, Any]]) -> Optional[str]:
    lines: List[str] = []
    for turn in conversation:
        u = (turn or {}).get("u") or ""
        a = (turn or {}).get("a") or ""
        if u:
            lines.append(f"U: {u}")
        if a:
            lines.append(f"A: {a}")
    return "\n".join(lines) if lines else None


def to_openeval(
    experiment_results: Any,
    logs: Iterable[Any],
    *,
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    experiment_id: Optional[str] = None,
    runner_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a humanbound experiment to an EvalPort ResultSet.

    `experiment_results` is a `humanbound.schemas.ExperimentResults` instance
    (or an equivalent dict with `stats`/`insights`/`posture`/`exec_t` keys) --
    the aggregate the experiment produced. `logs` is any iterable of
    `humanbound.schemas.LogEntry` instances (or equivalent dicts) -- the
    per-conversation verdicts. Both come straight from
    `humanbound.runner.LocalRunner.get_result()` / `.get_logs()`, or from
    `read_local_results()` / `local_results_to_openeval()` below.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if started_at is None:
        from datetime import datetime, timezone

        started_at = datetime.now(timezone.utc).isoformat()

    results: List[Dict[str, Any]] = []
    for i, log in enumerate(logs):
        log_dict = _to_dict(log) or {}
        thread_id = log_dict.get("thread_id") or f"log_{i}"
        verdict = (log_dict.get("result") or "").strip().lower()
        passed = verdict == "pass"

        conversation = list(log_dict.get("conversation") or [])
        actual_output = _render_transcript(conversation)

        exec_t = log_dict.get("exec_t") or 0  # seconds; 0/absent means "no timing recorded"
        duration_ms = int(round(exec_t * 1000)) if exec_t else None

        severity = log_dict.get("severity")
        confidence = log_dict.get("confidence")
        gen_category = log_dict.get("gen_category") or None
        fail_category = log_dict.get("fail_category") or None
        explanation = log_dict.get("explanation") or None

        score = None if verdict == "error" else _severity_score(severity if severity is not None else 0)

        grader_result: Dict[str, Any] = {
            "grader_id": "humanbound_verdict",
            "type": "custom",
            "score": score,
            "passed": passed,
            "metadata": {
                "gen_category": gen_category,
                "fail_category": fail_category,
                "severity": severity,
                "confidence": confidence,
            },
        }

        if explanation is not None:
            # reason is an optional string in resultset.json: omitted, not null,
            # when humanbound recorded no explanation (typical for a pass).
            grader_result["reason"] = explanation

        result: Dict[str, Any] = {
            "test_case_id": str(thread_id),
            "passed": passed,
            "grader_results": [grader_result],
        }
        if actual_output is not None:
            result["actual_output"] = actual_output
        if duration_ms is not None:
            result["duration_ms"] = duration_ms

        if verdict == "error":
            # error.type is a closed enum in resultset.json (timeout |
            # provider_error | runner_error). A judge failure is still told
            # apart by the grader's metadata.fail_category == "judge_error".
            result["error"] = {
                "type": "runner_error",
                "message": explanation or fail_category or "humanbound reported result=error with no explanation",
            }

        hb_meta: Dict[str, Any] = {}
        if conversation:
            hb_meta["conversation"] = conversation
        log_meta = log_dict.get("meta")
        if log_meta:
            hb_meta["log_meta"] = log_meta
        if hb_meta:
            result["metadata"] = {"humanbound": hb_meta}

        results.append(result)

    if not results:
        # EvalPort's spec requires a non-empty $.results
        # (openeval.validate.validate_result_set rejects `[]`) -- matches
        # humanbound's own presenter.run(), which treats zero logs as an
        # error state ("No logs to analyse.") rather than an empty success.
        # Raised here, at the point the caller can act on it, rather than
        # handed back as an unvalidatable ResultSet.
        raise ValueError(
            "to_openeval() received zero logs; EvalPort's ResultSet.results must be "
            "non-empty (spec/SPEC.md), and an experiment with no logs has nothing to report"
        )

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    rs_summary = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": (passed_count / total) if total else 0,
    }

    er_dict = _to_dict(experiment_results) or {}
    hb_rs_meta: Dict[str, Any] = {}
    if experiment_id:
        hb_rs_meta["experiment_id"] = experiment_id
    if er_dict.get("stats"):
        hb_rs_meta["stats"] = er_dict["stats"]
    if er_dict.get("posture"):
        hb_rs_meta["posture"] = er_dict["posture"]
    if er_dict.get("insights"):
        hb_rs_meta["insights"] = er_dict["insights"]
    if er_dict.get("exec_t"):
        hb_rs_meta["exec_t"] = er_dict["exec_t"]

    runner: Dict[str, str] = {"name": "humanbound"}
    if runner_version:
        runner["version"] = runner_version

    metadata: Dict[str, Any] = {"openeval": {"source": "humanbound"}}
    if hb_rs_meta:
        metadata["humanbound"] = hb_rs_meta

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at or started_at,
        "runner": runner,
        "results": results,
        "summary": rs_summary,
        "metadata": metadata,
    }


def from_openeval(result_set: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct humanbound-shaped experiment data from an EvalPort ResultSet.

    Returns `{"experiment_results": {...}, "logs": [...], "experiment_id": ...}`
    (the last key only when known). `logs` entries are plain dicts shaped
    like `humanbound.schemas.LogEntry` (`LogEntry(**log)` should construct
    directly when `humanbound` is installed).

    Fields this adapter itself wrote under `metadata.humanbound` (on either
    the `ResultSet` or an individual `Result`) round-trip exactly. For a
    `ResultSet` produced by another tool (no `metadata.humanbound.stats`),
    `experiment_results.stats` is recomputed from the reconstructed logs
    (pass/fail/error/total only) rather than left absent -- but `posture`,
    `insights`, and the aggregate `exec_t` are left `None`/empty rather than
    fabricated, since this adapter has no faithful way to derive humanbound's
    own posture/insight formulas (see `humanbound_cli/engine/presenter.py`)
    from `Result` data alone.
    """
    hb_rs_meta = (result_set.get("metadata") or {}).get("humanbound") or {}

    logs: List[Dict[str, Any]] = []
    for r in result_set.get("results", []):
        grader_results = r.get("grader_results") or []
        primary = grader_results[0] if grader_results else {}
        gmeta = primary.get("metadata") or {}
        hb_meta = (r.get("metadata") or {}).get("humanbound") or {}
        conversation = [
            {"u": (t or {}).get("u", ""), "a": (t or {}).get("a", "")}
            for t in (hb_meta.get("conversation") or [])
        ]

        if r.get("error"):
            result_label = "error"
        elif r.get("passed"):
            result_label = "pass"
        else:
            result_label = "fail"

        duration_ms = r.get("duration_ms")
        exec_t = (duration_ms / 1000.0) if duration_ms is not None else 0

        logs.append(
            {
                "thread_id": r.get("test_case_id", ""),
                "conversation": conversation,
                "result": result_label,
                "gen_category": gmeta.get("gen_category") or "",
                "fail_category": gmeta.get("fail_category") or "",
                "explanation": primary.get("reason") or "",
                "severity": gmeta.get("severity") if gmeta.get("severity") is not None else 0,
                "confidence": gmeta.get("confidence") if gmeta.get("confidence") is not None else 0,
                "exec_t": exec_t,
                "meta": hb_meta.get("log_meta") or {},
            }
        )

    if hb_rs_meta:
        stats = hb_rs_meta.get("stats")
        posture = hb_rs_meta.get("posture")
        insights = hb_rs_meta.get("insights") or []
        exec_t_agg = hb_rs_meta.get("exec_t") or {}
    else:
        total = len(logs)
        passed = sum(1 for l in logs if l["result"] == "pass")
        failed = sum(1 for l in logs if l["result"] == "fail")
        errors = sum(1 for l in logs if l["result"] == "error")
        stats = {"pass": passed, "fail": failed, "total": total, "error": errors, "unjudged": 0}
        posture = None
        insights = []
        exec_t_agg = {}

    out: Dict[str, Any] = {
        "experiment_results": {
            "stats": stats,
            "insights": insights,
            "posture": posture,
            "exec_t": exec_t_agg,
        },
        "logs": logs,
    }
    if "experiment_id" in hb_rs_meta:
        out["experiment_id"] = hb_rs_meta["experiment_id"]
    return out


def read_local_results(results_dir: PathLike) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[str]]:
    """Read a real `humanbound.runner.LocalRunner` result directory.

    `results_dir` is `.humanbound/results/<experiment_id>/`, exactly as
    written by `LocalTestRunner._save_results()`
    (`humanbound_cli/engine/local_runner.py`): `meta.json` (an
    `ExperimentMeta`, validated Pydantic; its `.results` field holds the
    `ExperimentResults`) and `logs.jsonl` (one validated `LogEntry` JSON
    object per line).

    Returns `(experiment_results, logs, experiment_id)`, ready to pass
    straight into `to_openeval()` -- or use `local_results_to_openeval()`
    below to skip the intermediate step.
    """
    results_dir = Path(results_dir)
    meta_path = results_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no meta.json in {results_dir} -- expected a humanbound LocalRunner result "
            "directory (.humanbound/results/<experiment_id>/)"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    experiment_results = meta.get("results") or {}
    experiment_id = meta.get("id")

    logs_path = results_dir / "logs.jsonl"
    logs: List[Dict[str, Any]] = []
    if logs_path.exists():
        for line in logs_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    return experiment_results, logs, experiment_id


def local_results_to_openeval(results_dir: PathLike, *, suite_id: str, run_id: str, **kwargs: Any) -> Dict[str, Any]:
    """`read_local_results()` + `to_openeval()` in one call.

    `experiment_id` (read from `meta.json`) is passed through to
    `to_openeval()` automatically unless overridden in `kwargs`.
    """
    experiment_results, logs, experiment_id = read_local_results(results_dir)
    kwargs.setdefault("experiment_id", experiment_id)
    return to_openeval(experiment_results, logs, suite_id=suite_id, run_id=run_id, **kwargs)
