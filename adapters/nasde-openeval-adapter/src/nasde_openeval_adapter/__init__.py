"""nasde-toolkit <-> EvalPort adapter.

Converts the flat, per-trial output of `nasde results-export` (a NoesisVision
nasde-toolkit CLI command; https://github.com/NoesisVision/nasde-toolkit) into
the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside nasde-toolkit
itself: the maintainer explicitly declined an EvalPort dependency in nasde's
core (nasde has zero optional extras -- everything in `[project.dependencies]`
ships to every install, so an interchange-format dependency used by only some
users is a cost the maintainer doesn't want every install to pay), but pointed
to `nasde results-export` as a stable-enough surface to build an external
converter against. See the full design discussion at
https://github.com/NoesisVision/nasde-toolkit/issues/79.

This adapter reads ONLY the files `nasde results-export JOB_DIR --to DEST`
writes into each `DEST/<job>__<trial>/` directory:

- metrics.json          (trial_name, task_name, agent_name, model_name,
                          reasoning_effort, source, started_at, finished_at,
                          duration_sec, harbor_reward, score, score_eval_std,
                          score_eval_n, single_eval, token_usage, cost_usd,
                          pricing_as_of, exception_info)
- assessment_summary.json (AssessmentSummary: task_name, trial_name,
                          agent_name, groups[] of EvaluatorGroupSummary, each
                          with evaluator_model, dimensions_fingerprint, n,
                          dominant, dimensions[] of DimensionStats
                          (name, max_score, mean, std, min, max),
                          normalized_score_mean/std/min/max, total_score_mean,
                          harbor_reward, duration_sec_mean)
- assessment_eval_<N>.json (per-repetition EvaluationResult, used only to
                          recover a free-text `summary` for `actual_output`
                          when the dominant cluster has one)

It never reads nasde internals directly (no imports from nasde_toolkit, no
parsing of Harbor's raw jobs/<job>/<task>__<id>/ trial layout) -- exactly the
"nothing from nasde internals" boundary the maintainer asked for, since
`~/.cannonade/*`-style internal layouts are unpublished and can change without
notice, while `results-export`'s output is the documented, if still
experimental, surface.

## Design decisions that respond directly to the maintainer's three stated
## requirements (quoted from issue #79)

1. "`passed = harbor_reward >= 1.0` throws away the main result... The primary
   score is a multi-dimension LLM-judge rubric... `grader_results` looks like
   the right place for the dimensions, with the score rescaled to 0..1 and the
   original scale kept in metadata. The required `passed` boolean per grader
   is the awkward part, since we don't define pass thresholds per dimension."

   -> Each `DimensionStats` entry from the *dominant* `EvaluatorGroupSummary`
   becomes one `GraderResult` (`grader_id="dim_<name>"`, `type="llm_judge"`),
   with `score = mean / max_score` (rescaled into EvalPort's required [0,1]
   range) and `mean`/`std`/`min`/`max`/`max_score` preserved verbatim in
   `metadata`. nasde does not define a per-dimension pass threshold, so this
   adapter does NOT invent one silently: `passed` per dimension is
   `score >= pass_threshold`, where `pass_threshold` is an explicit,
   documented constructor parameter (default 0.5) that callers can and should
   override to match their own bar. `harbor_reward` is preserved too, but as
   its OWN separate `grader_id="harbor_reward"` GraderResult (type="custom",
   params.handler="nasde_harbor_verifier") rather than folding it into the
   overall `Result.passed` -- it is a secondary, binary verifier signal, not
   the primary rubric score.

   `Result.passed` itself is `score_stats["score"] >= pass_threshold` (the
   dominant cluster's `normalized_score_mean`, i.e. exactly what
   `metrics.json`'s top-level `score` field already is per nasde's own
   `_resolve_score_stats()` -- NOT a re-derivation from `harbor_reward`).
   Only when no assessment ran at all (no `assessment_summary.json`, e.g. the
   agent crashed) does this adapter fall back to
   `harbor_reward >= reward_fallback_threshold` (default 1.0) as the sole
   available signal -- and that fallback is called out explicitly in
   `Result.metadata["nasde"]["passed_basis"]` so it is never confused with a
   real judge verdict.

2. "Each trial is judged N times... aggregated... only within one
   (evaluator_model, dimensions_fingerprint) cluster... the cluster key,
   `eval_n` and the std would have to go into metadata... worth saying in the
   adapter README so nobody averages across clusters downstream."

   -> `Result.metadata["nasde"]["dominant_cluster"]` records
   `evaluator_model`, `dimensions_fingerprint`, and `eval_n` (the cluster's
   `n`) for the cluster the `grader_results` scores came from.  Every
   *non*-dominant cluster (a different judge model or a changed rubric) is
   preserved too -- never dropped -- as a separate list at
   `Result.metadata["nasde"]["non_dominant_clusters"]`, each with its own
   `evaluator_model`/`dimensions_fingerprint`/`eval_n`/`normalized_score_mean`.
   **These are archival only.** See the "Cluster safety" section below and in
   the README: nothing in this adapter, and nothing downstream, should ever
   average or otherwise mix `grader_results` scores (from the dominant
   cluster) with anything under `non_dominant_clusters` -- they are different
   benchmarks by nasde's own definition.

3. "Per-trial economics (token_usage, cost_usd, pricing_as_of, model_name,
   reasoning_effort) are what we plot... if those get dropped the export
   isn't usable for our main case."

   -> All five fields are preserved verbatim, unmodified, in
   `Result.metadata["nasde"]["economics"]`
   (`token_usage`, `cost_usd`, `pricing_as_of`, `model_name`,
   `reasoning_effort`), read straight from `metrics.json`. `model_name` is
   also mirrored into `Result.metadata["nasde"]["model_name"]` at the top
   level for convenience since it is also a natural grouping key, but the
   canonical values live in the `economics` block untouched.

## Cluster safety

Nothing in `to_openeval()` ever combines two different
`(evaluator_model, dimensions_fingerprint)` clusters into one score. Only the
dominant cluster (the one nasde itself marks `dominant: true`, or the sole
cluster if `assessment_summary.json` has just one) feeds `grader_results`.
Every consumer of this adapter's output MUST preserve that boundary: if you
build any rollup across multiple `Result`s, only ever compare/average
`grader_results` scores whose `metadata["nasde"]["dominant_cluster"]` shares
the same `(evaluator_model, dimensions_fingerprint)` pair. Mixing clusters
silently produces a meaningless number, per the maintainer's own stated
concern -- this is not a hypothetical, it is the literal thing issue #79
asked this adapter to avoid.

## Lossiness of from_openeval()

`from_openeval()` is necessarily lossy and best-effort: EvalPort's `Result`
is an aggregate view, while nasde's own trial directory carries per-repetition
`assessment_eval_<N>.json` files, a full agent `trajectory.json`, a
`changes.patch`, and raw verifier stdout that no aggregate format can
reconstruct. It returns the flattened, `metrics.json`-shaped view only --
never claim it reconstructs a full nasde trial directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0-rc.5"

__all__ = ["to_openeval", "from_openeval", "trial_to_result", "__version__"]
__version__ = "0.1.0"

DEFAULT_PASS_THRESHOLD = 0.5
DEFAULT_REWARD_FALLBACK_THRESHOLD = 1.0

_ECONOMICS_FIELDS = ("token_usage", "cost_usd", "pricing_as_of", "model_name", "reasoning_effort")


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    return data


def _dominant_group(groups: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not groups:
        return None
    for g in groups:
        if g.get("dominant"):
            return g
    return groups[0]


def _clip_unit(value: Optional[float]) -> Optional[float]:
    """Clip a score into EvalPort's required [0, 1] grader-score range.

    nasde's harbor_reward is documented as a verifier reward and is 0/1 or a
    fractional partial-credit value in every real case observed, but nothing
    in nasde's schema *guarantees* it stays inside [0, 1] forever. Rather than
    let an out-of-range value silently fail validate_result_set() (or worse,
    pass through into `passed` math wrong), this clips defensively and keeps
    the untouched raw value in metadata so nothing is silently lost.
    """
    if value is None:
        return None
    return max(0.0, min(1.0, float(value)))


def _resolve_actual_output(trial_dir: Path, dominant: Optional[Dict[str, Any]]) -> Optional[str]:
    """Best-effort: pull the dominant cluster's most recent judge free-text summary.

    Each assessment_eval_<N>.json (an EvaluationResult) carries its own
    evaluator_model, dimensions_fingerprint, timestamp and summary. If the
    dominant cluster's key matches one or more of these files, the most
    recent (by timestamp) summary is used as `actual_output`. Returns None
    when no matching file exists -- actual_output is optional in the spec and
    this adapter never fabricates prose that didn't come from a real judge.
    """
    if dominant is None:
        return None
    key = (dominant.get("evaluator_model"), dominant.get("dimensions_fingerprint"))
    candidates = []
    for eval_path in sorted(trial_dir.glob("assessment_eval_*.json")):
        try:
            data = _load_json(eval_path)
        except (json.JSONDecodeError, OSError):
            continue
        if (data.get("evaluator_model"), data.get("dimensions_fingerprint")) == key:
            candidates.append(data)
    if not candidates:
        return None
    candidates.sort(key=lambda d: d.get("timestamp", ""))
    summary = candidates[-1].get("summary")
    return summary or None


def trial_to_result(
    trial_dir: Union[str, Path],
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    reward_fallback_threshold: float = DEFAULT_REWARD_FALLBACK_THRESHOLD,
) -> Dict[str, Any]:
    """Convert one `nasde results-export` trial directory into one EvalPort Result dict.

    `trial_dir` is one `<job>__<trial>/` directory as written by
    `nasde results-export JOB_DIR --to DEST` (i.e. one entry directly under
    DEST). Raises FileNotFoundError if `metrics.json` is missing -- every
    trial directory results-export writes always has one, so its absence
    means `trial_dir` isn't really a results-export output directory.
    """
    trial_dir = Path(trial_dir)
    metrics_path = trial_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{trial_dir} has no metrics.json -- this doesn't look like a "
            "`nasde results-export` output directory. Point trial_to_result() "
            "at one of the <job>__<trial>/ subdirectories results-export writes, "
            "not at a Harbor jobs/ tree directly."
        )
    metrics = _load_json(metrics_path)

    summary_path = trial_dir / "assessment_summary.json"
    groups: List[Dict[str, Any]] = []
    if summary_path.exists():
        groups = _load_json(summary_path).get("groups", [])
    dominant = _dominant_group(groups)
    non_dominant = [g for g in groups if g is not dominant]

    grader_results: List[Dict[str, Any]] = []
    passed_basis = "no_assessment"

    if dominant is not None:
        for dim in dominant.get("dimensions", []):
            max_score = dim.get("max_score") or 0
            mean = dim.get("mean", 0.0)
            rescaled = _clip_unit(mean / max_score) if max_score else None
            grader_results.append(
                {
                    "grader_id": f"dim_{dim['name']}",
                    "type": "llm_judge",
                    "score": rescaled,
                    "passed": bool(rescaled is not None and rescaled >= pass_threshold),
                    "reason": (
                        f"mean {mean} +/- {dim.get('std', 0.0)} over n={dominant.get('n')} "
                        f"({dominant.get('evaluator_model')}), scale 0-{max_score}"
                    ),
                    "metadata": {
                        "dimension_name": dim["name"],
                        "mean": mean,
                        "std": dim.get("std"),
                        "min": dim.get("min"),
                        "max": dim.get("max"),
                        "max_score": max_score,
                        "evaluator_model": dominant.get("evaluator_model"),
                        "dimensions_fingerprint": dominant.get("dimensions_fingerprint"),
                        "eval_n": dominant.get("n"),
                        "dominant": True,
                    },
                }
            )
        passed_basis = "dominant_cluster_normalized_score"

    harbor_reward_raw = metrics.get("harbor_reward")
    if harbor_reward_raw is not None:
        harbor_score = _clip_unit(harbor_reward_raw)
        grader_results.append(
            {
                "grader_id": "harbor_reward",
                "type": "custom",
                "score": harbor_score,
                "passed": bool(harbor_score is not None and harbor_score >= 1.0),
                "reason": "Harbor verifier reward (secondary, binary/partial-credit verifier signal)",
                "metadata": {
                    "handler": "nasde_harbor_verifier",
                    "raw_harbor_reward": harbor_reward_raw,
                },
            }
        )

    score_stats_score = metrics.get("score")
    if score_stats_score is not None:
        overall_passed = bool(score_stats_score >= pass_threshold)
    elif harbor_reward_raw is not None:
        overall_passed = bool(harbor_reward_raw >= reward_fallback_threshold)
        passed_basis = "harbor_reward_fallback_no_judge_assessment"
    else:
        overall_passed = False
        passed_basis = "no_signal_available"

    duration_sec = metrics.get("duration_sec")
    duration_ms = int(round(duration_sec * 1000)) if duration_sec is not None else None

    error = None
    if metrics.get("exception_info"):
        error = {"type": "agent_exception", "detail": metrics["exception_info"]}

    nasde_meta: Dict[str, Any] = {
        "passed_basis": passed_basis,
        "trial_name": metrics.get("trial_name"),
        "task_name": metrics.get("task_name"),
        "agent_name": metrics.get("agent_name"),
        "model_name": metrics.get("model_name"),
        "source": metrics.get("source"),
        "harbor_reward": harbor_reward_raw,
        "score_eval_std": metrics.get("score_eval_std"),
        "score_eval_n": metrics.get("score_eval_n"),
        "single_eval": metrics.get("single_eval"),
        "economics": {field: metrics.get(field) for field in _ECONOMICS_FIELDS},
    }
    if dominant is not None:
        nasde_meta["dominant_cluster"] = {
            "evaluator_model": dominant.get("evaluator_model"),
            "dimensions_fingerprint": dominant.get("dimensions_fingerprint"),
            "eval_n": dominant.get("n"),
            "normalized_score_mean": dominant.get("normalized_score_mean"),
            "normalized_score_std": dominant.get("normalized_score_std"),
        }
    if non_dominant:
        # Archival only -- see module docstring "Cluster safety". Never averaged
        # with grader_results, which reflect the dominant cluster exclusively.
        nasde_meta["non_dominant_clusters"] = [
            {
                "evaluator_model": g.get("evaluator_model"),
                "dimensions_fingerprint": g.get("dimensions_fingerprint"),
                "eval_n": g.get("n"),
                "normalized_score_mean": g.get("normalized_score_mean"),
                "normalized_score_std": g.get("normalized_score_std"),
                "dimensions": g.get("dimensions"),
            }
            for g in non_dominant
        ]

    result: Dict[str, Any] = {
        "test_case_id": metrics.get("task_name") or metrics.get("trial_name") or trial_dir.name,
        "passed": overall_passed,
        "grader_results": grader_results,
        "actual_output": _resolve_actual_output(trial_dir, dominant),
        "duration_ms": duration_ms,
        "completed_at": metrics.get("finished_at") or None,
        "metadata": {"nasde": nasde_meta},
    }
    if error is not None:
        result["error"] = error
    return result


def to_openeval(
    trial_dirs: Sequence[Union[str, Path]],
    *,
    run_id: Optional[str] = None,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    reward_fallback_threshold: float = DEFAULT_REWARD_FALLBACK_THRESHOLD,
) -> Dict[str, Any]:
    """Convert a list of `nasde results-export` trial directories into one EvalPort ResultSet.

    `trial_dirs` is the list of `<job>__<trial>/` directories written by one
    `nasde results-export JOB_DIR --to DEST` invocation (or a hand-picked
    subset of them) -- pass `sorted(Path(dest).iterdir())` for "everything
    exported to DEST". All trials must share the same `source` (nasde's
    benchmark/challenge-set identifier from `metrics.json`); this becomes
    `ResultSet.suite_id`. Call `to_openeval()` once per distinct `source` if
    an export directory mixes benchmarks -- this adapter refuses to guess a
    suite_id spanning two different benchmarks.

    `run_id` defaults to the shared `<job>` prefix parsed from each
    directory's `<job>__<trial>` name; pass it explicitly if trial_dirs
    doesn't follow that naming convention (e.g. directories were renamed) or
    spans more than one job.

    Returns a plain dict conforming to EvalPort's ResultSet schema. Validate
    it with `openeval.validate.validate_result_set()` before use.
    """
    trial_dirs = [Path(d) for d in trial_dirs]
    if not trial_dirs:
        raise ValueError("to_openeval() requires at least one trial directory")

    results = [
        trial_to_result(
            d, pass_threshold=pass_threshold, reward_fallback_threshold=reward_fallback_threshold
        )
        for d in trial_dirs
    ]

    sources = set()
    for d in trial_dirs:
        metrics = _load_json(d / "metrics.json")
        sources.add(metrics.get("source") or "")
    if len(sources) > 1:
        raise ValueError(
            f"trial_dirs spans multiple nasde 'source' benchmarks ({sorted(sources)}) -- "
            "call to_openeval() separately per source; a single ResultSet must "
            "represent one benchmark/suite, not a mix."
        )
    suite_id = (sources.pop() if sources else "") or "nasde_unspecified_source"

    if run_id is None:
        job_names = set()
        for d in trial_dirs:
            job_names.add(d.name.split("__", 1)[0] if "__" in d.name else d.name)
        if len(job_names) == 1:
            run_id = job_names.pop()
        else:
            raise ValueError(
                f"trial_dirs spans multiple job names ({sorted(job_names)}) and no "
                "explicit run_id was given -- pass run_id= explicitly, since a "
                "ResultSet.run_id must identify one execution/run."
            )

    started_candidates = []
    finished_candidates = []
    for d in trial_dirs:
        metrics = _load_json(d / "metrics.json")
        if metrics.get("started_at"):
            started_candidates.append(metrics["started_at"])
        if metrics.get("finished_at"):
            finished_candidates.append(metrics["finished_at"])
    started_at = min(started_candidates) if started_candidates else ""
    completed_at = max(finished_candidates) if finished_candidates else None

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "runner": {"name": "nasde-toolkit", "version": "results-export"},
        "results": results,
        "metadata": {
            "openeval": {"source": "nasde-toolkit"},
            "nasde": {
                "note": (
                    "grader_results reflect only each trial's dominant "
                    "(evaluator_model, dimensions_fingerprint) cluster; see "
                    "each Result.metadata.nasde.non_dominant_clusters for "
                    "archival, never-averaged alternate-cluster data."
                ),
            },
        },
    }


def from_openeval(result_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort, lossy import of an EvalPort ResultSet back to nasde-shaped dicts.

    Returns one flattened, metrics.json-shaped dict per Result (trial_name,
    task_name, model_name, harbor_reward, score, duration_sec, economics
    fields, ...), reconstructed from grader_results/metadata where this
    adapter itself put them. This can NOT reconstruct a real nasde trial
    directory: there is no way back to per-repetition assessment_eval_N.json
    files, the agent trajectory, changes.patch, or verifier stdout from an
    aggregated Result alone. Use this only to round-trip the summary-level
    fields this adapter itself emits (e.g. for testing), not as a general
    EvalPort -> nasde importer for arbitrary producers' ResultSets.
    """
    out: List[Dict[str, Any]] = []
    for result in result_set.get("results", []):
        nasde_meta = (result.get("metadata") or {}).get("nasde", {})
        economics = nasde_meta.get("economics", {}) or {}
        duration_ms = result.get("duration_ms")
        out.append(
            {
                "trial_name": nasde_meta.get("trial_name") or result.get("test_case_id"),
                "task_name": nasde_meta.get("task_name") or result.get("test_case_id"),
                "agent_name": nasde_meta.get("agent_name"),
                "model_name": economics.get("model_name") or nasde_meta.get("model_name"),
                "reasoning_effort": economics.get("reasoning_effort"),
                "source": nasde_meta.get("source"),
                "finished_at": result.get("completed_at"),
                "duration_sec": (duration_ms / 1000.0) if duration_ms is not None else None,
                "harbor_reward": nasde_meta.get("harbor_reward"),
                "score": (nasde_meta.get("dominant_cluster") or {}).get("normalized_score_mean"),
                "score_eval_std": nasde_meta.get("score_eval_std"),
                "score_eval_n": nasde_meta.get("score_eval_n"),
                "single_eval": nasde_meta.get("single_eval"),
                "token_usage": economics.get("token_usage"),
                "cost_usd": economics.get("cost_usd"),
                "pricing_as_of": economics.get("pricing_as_of"),
                "exception_info": (result.get("error") or {}).get("detail")
                if result.get("error")
                else None,
            }
        )
    return out
