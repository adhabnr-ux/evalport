"""Convert FrontierOR (https://github.com/Minw913/FrontierOR) evaluation
results and public task instances to the EvalPort open evaluation format.

Built at the FrontierOR maintainer's request in Minw913/FrontierOR#5, against
the repository at commit ``bc1bdde`` and the ``SmartOR/FrontierOR`` Hugging
Face dataset v1 update (2026-09-27). Everything below was read from that code,
not guessed:

- ``trusted_eval_infra/contracts.py``: ``SCORING_CONTRACT_VERSION =
  "staged-qte-v1"``, ``public_scoring_contract()`` (the staged_qte equation,
  the scale ``D``, 6-decimal rounding, mean aggregation with missing/failed
  instances scored 0.0) and ``visibility_contract()`` (reference objectives,
  reference runtimes and final per-instance scores are ``TRUSTED_ONLY``).
- ``test_time_self_evolution/scoring/staged_qte.py``: ``StagedQteScorer``.
- ``test_time_self_evolution/eval_modes.py``:
  ``augment_results_with_staged_qte()``, whose per-instance output fields
  (``score``, ``stage_id``, ``quality_part``, ``speed_part``, ``signed_gap``,
  ``beat_amount``, ``beat_gurobi_flag``, ``matched_flag``) are consumed as-is.
- ``one_shot_eval.py``: ``RESULTS_CSV_COLUMNS`` and ``compute_gap()``.
- ``scripts/compute_benchmark_main_metrics.py``: the paper's binary QTE,
  ``beat_gurobi_1``.

Three graders are emitted per result, and they measure different things:

``gr_frontieror_quality_only`` (``custom``, handler ``frontieror:quality_only``)
    ``min(1, max(0, 1 - g))`` on the contract's SIGNED gap ``g``. This is NOT
    staged_qte: it drops the speed term and clamps both sides so it fits
    EvalPort's [0, 1] score slot. The lower clamp is needed because ``g`` can
    exceed 1; the upper clamp because ``g`` is negative when the candidate
    beats the reference. ``score`` is null when ``g`` is undefined
    (infeasible, not run, missing objective or reference).

``gr_frontieror_staged_qte`` (``custom``, handler ``frontieror:staged_qte``)
    FrontierOR's own score, carried unchanged. staged_qte ranges over
    ``[0, 2 + beat_amount]`` and has no upper bound, so it cannot go in
    EvalPort's [0, 1] ``score`` without being clamped or rescaled. Either one
    would change the number. ``score`` is therefore always null, and so
    ``passed`` is always false: EvalPort Validation Rule 6 says a null score
    means "not verified" and MUST carry ``passed: false``. Whether the result
    reached stage 2 is ``metadata.frontieror.stage_id``. The raw
    value plus every debug field (``stage_id``, ``quality_part``,
    ``speed_part``, ``signed_gap``, ``beat_amount``, ``matched``,
    ``beat_gurobi``, the contract version and the scorer source) live in
    ``metadata.frontieror``.

``gr_frontieror_binary_qte`` (``custom``, handler ``frontieror:beat_gurobi_1``)
    The paper's binary QTE: feasible AND gap <= 1% AND budget-capped wall
    time no slower than Gurobi's within the clock tolerance. Score 1.0 or 0.0.
    ``Result.passed`` is this grader's verdict.

Visibility: the adapter exports reference objectives and runtimes only for
instances found in the PUBLIC reference table the caller passes
(``metadata/gurobi_references.{parquet,csv.gz}`` from the Hugging Face
dataset). Any result for an instance that is not in that table (for example a
hidden final instance from ``trusted_eval_infra``) makes the conversion fail by
default, or is dropped with ``on_non_public="omit"``. Only the number dropped
is recorded, never the ids, because final instance membership is also
``TRUSTED_ONLY``.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk is a hard dependency
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "results_to_openeval",
    "signed_gap",
    "quality_only_score",
    "staged_qte_builtin",
    "binary_qte",
    "load_references",
    "load_paper_meta",
    "load_dataset_instances",
    "read_results_csv",
    "split_by_model",
    "augmented_results_to_rows",
    "NonPublicInstanceError",
    "SCORING_CONTRACT_VERSION",
    "__version__",
]
__version__ = "0.1.0"

_NS = "frontieror"

# --- trusted_eval_infra/contracts.py (pinned; checked against an upstream
# checkout at runtime when one is importable) --------------------------------
SCORING_CONTRACT_VERSION = "staged-qte-v1"
CONTRACT_SCHEMA_VERSION = 1
STAGE_BOUNDARY = 0.01
NEAR_ZERO_REFERENCE = 1e-3
INSTANCE_SCORE_DECIMALS = 6

# --- scripts/compute_benchmark_main_metrics.py (binary QTE) -----------------
BINARY_QTE_GAP_THRESHOLD = 0.01  # beat_gurobi_1
T_MAX_LARGE = 3600.0
DEFAULT_QTE_TIME_TOLERANCE_SECONDS = 0.01
DEFAULT_QTE_TIME_TOLERANCE_FRACTION = 0.001
LARGE_INSTANCES: Tuple[str, ...] = ("large_1", "large_2", "large_3", "large_4", "large_5")
PUBLIC_INSTANCE_NAMES: Tuple[str, ...] = ("tiny",) + LARGE_INSTANCES

# Copied verbatim from compute_benchmark_main_metrics.py. Cells whose Gurobi
# reference is a checker-feasible, proven optimal zero: quality there is an
# absolute comparison |obj| <= tol instead of a relative gap.
PROVEN_ZERO_OPTIMUM_TOLERANCE: Dict[Tuple[str, str], float] = {
    **{("araujo2020", f"large_{i}"): 0.5 for i in (1, 3, 5)},
    **{("chen1999", f"large_{i}"): 0.5 for i in (4, 5)},
    **{("dienstknecht2024", f"large_{i}"): 0.5 for i in range(1, 6)},
    **{("earl2005", f"large_{i}"): 1e-6 for i in range(1, 6)},
    ("forrest2006", "large_1"): 1e-3,
    **{("frey2017", f"large_{i}"): 1e-3 for i in range(1, 6)},
    ("kowalczyk2024", "large_2"): 0.5,
    **{("oliveira2020", f"large_{i}"): 1e-5 for i in range(1, 6)},
    **{("wangk2020", f"large_{i}"): 1e-3 for i in range(1, 6)},
}

GRADER_QUALITY_ONLY = "gr_frontieror_quality_only"
GRADER_STAGED_QTE = "gr_frontieror_staged_qte"
GRADER_BINARY_QTE = "gr_frontieror_binary_qte"


def _graders(stage_boundary: float) -> List[Dict[str, Any]]:
    return [
        {
            "id": GRADER_QUALITY_ONLY,
            "type": "custom",
            "description": (
                "Normalized quality-only score min(1, max(0, 1 - g)) on the "
                "staged-qte-v1 signed gap g. NOT FrontierOR's staged_qte: the "
                "speed term is dropped and both sides are clamped to fit [0, 1]."
            ),
            "params": {
                "handler": "frontieror:quality_only",
                "formula": "min(1, max(0, 1 - g))",
                "signed_gap": "g = (c - r)/D (min) or (r - c)/D (max); D = abs(r) if abs(r) >= 0.001 else max(abs(c), 0.001)",
                "pass_rule": "feasible and max(0, g) <= stage_boundary",
                "stage_boundary": stage_boundary,
                "is_staged_qte": False,
                "undefined_score": "null (infeasible, not run, or missing objective/reference)",
            },
        },
        {
            "id": GRADER_STAGED_QTE,
            "type": "custom",
            "description": (
                "FrontierOR staged_qte (staged-qte-v1), unchanged. Unbounded "
                "above ([0, 2 + beat_amount]), so the EvalPort score slot is "
                "always null and the raw value is in metadata.frontieror.staged_qte."
            ),
            "params": {
                "handler": "frontieror:staged_qte",
                "contract_version": SCORING_CONTRACT_VERSION,
                "stage_boundary": stage_boundary,
                "stage_1": "max(0, 1 - g) when max(0, g) > b",
                "stage_2": "(1 - g) + max(0, 1 - t/tau) when max(0, g) <= b",
                "score_slot": "null; raw value in metadata.frontieror.staged_qte",
                "pass_rule": (
                    "always false: score is null (not verified), and EvalPort Validation "
                    "Rule 6 requires passed false with a null score. The stage reached "
                    "is metadata.frontieror.stage_id."
                ),
            },
        },
        {
            "id": GRADER_BINARY_QTE,
            "type": "custom",
            "description": (
                "The FrontierOR paper's binary QTE (beat_gurobi_1): feasible, "
                "gap <= 1%, and budget-capped wall time no slower than Gurobi "
                "within the clock tolerance."
            ),
            "params": {
                "handler": "frontieror:beat_gurobi_1",
                "gap_threshold": BINARY_QTE_GAP_THRESHOLD,
                "time_tolerance_seconds": DEFAULT_QTE_TIME_TOLERANCE_SECONDS,
                "time_tolerance_fraction": DEFAULT_QTE_TIME_TOLERANCE_FRACTION,
                "candidate_time_cap_seconds_large": T_MAX_LARGE,
                "source": "scripts/compute_benchmark_main_metrics.py",
            },
        },
    ]


class NonPublicInstanceError(ValueError):
    """Raised when a result refers to an instance that is not in the public
    reference table, so exporting it could leak TRUSTED_ONLY data."""


# ---------------------------------------------------------------------------
# small parsing helpers
# ---------------------------------------------------------------------------

def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ("none", "nan", "null"):
            return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _to_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "1", "1.0", "yes"):
        return True
    if text in ("false", "0", "0.0", "no"):
        return False
    return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if text.strip() != "" else None


# ---------------------------------------------------------------------------
# scoring primitives
# ---------------------------------------------------------------------------

def signed_gap(candidate_objective: float, reference_objective: float, direction: str) -> float:
    """staged-qte-v1 signed relative gap: ``g = (c-r)/D`` for minimization,
    ``(r-c)/D`` for maximization, ``D = abs(r) if abs(r) >= 0.001 else
    max(abs(c), 0.001)``. Negative means the candidate beats the reference.
    Same arithmetic as ``staged_qte.signed_quality_gap``."""
    if direction not in ("min", "max"):
        raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
    c = float(candidate_objective)
    r = float(reference_objective)
    denom = abs(r) if abs(r) >= NEAR_ZERO_REFERENCE else max(abs(c), NEAR_ZERO_REFERENCE)
    denom = max(denom, 1e-10)
    sign = 1 if direction == "min" else -1
    return sign * (c - r) / denom


def quality_only_score(g: float) -> float:
    """``min(1, max(0, 1 - g))``, rounded like the contract (6 decimals).

    Two-sided on purpose: ``max(0, .)`` alone is still unbounded above,
    because ``g < 0`` whenever the candidate beats the reference."""
    return round(min(1.0, max(0.0, 1.0 - float(g))), INSTANCE_SCORE_DECIMALS)


def _effective_runtime(runtime: Any, time_limit: Any) -> Optional[float]:
    """``building_blocks.effective_runtime``: wall time capped by its budget."""
    try:
        runtime_value = float(runtime)
        limit_value = float(time_limit)
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not math.isfinite(runtime_value)
        or not math.isfinite(limit_value)
        or runtime_value < 0
        or limit_value <= 0
    ):
        return None
    return min(runtime_value, limit_value)


def _empty_debug(reason: str) -> Dict[str, Any]:
    return {
        "stage_id": 0,
        "quality_part": 0.0,
        "speed_part": 0.0,
        "signed_gap": 1.0,
        "beat_amount": 0.0,
        "matched": False,
        "beat_gurobi": False,
        "reason": reason,
    }


def staged_qte_builtin(
    result: Mapping[str, Any],
    *,
    time_limit: float,
    gurobi_obj: Optional[float],
    gurobi_time: Optional[float],
    gurobi_time_limit: Optional[float],
    direction: str,
    stage_boundary: float = STAGE_BOUNDARY,
) -> Tuple[float, Dict[str, Any]]:
    """Line-for-line port of ``StagedQteScorer.score_instance`` (staged-qte-v1),
    used only when no FrontierOR checkout is importable and the input carries
    no precomputed score. ``result`` uses the scorer's own keys:
    ``feasible``, ``llm_obj``, ``solve_time``. The test suite checks it
    against the real scorer whenever a checkout is available."""
    if result.get("feasible") is not True:
        return 0.0, _empty_debug("infeasible")
    if gurobi_obj is None:
        return 0.0, _empty_debug("missing_gurobi_baseline")
    obj_llm = result.get("llm_obj")
    if obj_llm is None:
        return 0.0, _empty_debug("missing_llm_obj")

    g = signed_gap(float(obj_llm), float(gurobi_obj), direction)
    quality = 1.0 - g
    g_pos = max(0.0, g)

    if g_pos > stage_boundary:
        score = max(0.0, quality)
        stage_id = 1
        quality_part = score
        speed_part = 0.0
    else:
        t_solve_raw = result.get("solve_time")
        tau_g = gurobi_time
        try:
            t_solve = float(t_solve_raw) if t_solve_raw is not None else float(time_limit)
        except (TypeError, ValueError):
            t_solve = float(time_limit)
        if tau_g is None or tau_g <= 0:
            speed_part = 0.0
        else:
            candidate_time = _effective_runtime(t_solve, time_limit)
            gurobi_limit = gurobi_time_limit or time_limit
            gurobi_time_eff = _effective_runtime(tau_g, gurobi_limit)
            if candidate_time is None or gurobi_time_eff is None or gurobi_time_eff <= 0:
                speed_part = 0.0
            else:
                speed_part = max(0.0, 1.0 - candidate_time / gurobi_time_eff)
        quality_part = quality
        score = quality_part + speed_part
        stage_id = 2

    return round(score, 6), {
        "stage_id": stage_id,
        "quality_part": round(quality_part, 6),
        "speed_part": round(speed_part, 6),
        "signed_gap": round(g, 6),
        "beat_amount": round(max(0.0, -g), 6),
        "matched": bool(g_pos < 1e-4),
        "beat_gurobi": bool(g < -1e-4),
    }


def _oneshot_gap(llm_obj: Optional[float], gurobi_obj: Optional[float], direction: str) -> Optional[float]:
    """``one_shot_eval.compute_gap`` verbatim: the gap the paper's binary QTE
    uses. Denominator ``abs(r)`` (no near-zero scale), ``None`` when
    ``abs(r) < 1e-10`` and the values differ."""
    if llm_obj is None or gurobi_obj is None:
        return None
    if abs(gurobi_obj) < 1e-10:
        return 0.0 if abs(llm_obj) < 1e-10 else None
    if direction == "max":
        return (gurobi_obj - llm_obj) / abs(gurobi_obj)
    return (llm_obj - gurobi_obj) / abs(gurobi_obj)


def binary_qte(
    *,
    paper_id: str,
    instance: str,
    feasible: Optional[bool],
    candidate_objective: Optional[float],
    candidate_time: Optional[float],
    gap: Optional[float],
    reference_runtime: Optional[float],
    reference_time_limit: Optional[float],
    candidate_time_cap: float = T_MAX_LARGE,
    gap_threshold: float = BINARY_QTE_GAP_THRESHOLD,
    tolerance_seconds: float = DEFAULT_QTE_TIME_TOLERANCE_SECONDS,
    tolerance_fraction: float = DEFAULT_QTE_TIME_TOLERANCE_FRACTION,
) -> Tuple[bool, Dict[str, Any]]:
    """Per-cell ``beat_gurobi_1`` from ``compute_metrics()``: the pandas
    column logic evaluated for one cell, including its NaN handling (a
    missing Gurobi runtime falls back to the Gurobi time limit, a zero
    candidate time never counts as fast enough). ``gap`` is the one-shot
    CSV gap (``compute_gap``), not the staged_qte ``D``-scaled gap."""
    if tolerance_seconds < 0 or tolerance_fraction < 0:
        raise ValueError("QTE time tolerances must be non-negative")
    is_feasible = feasible is True
    zero_tol = PROVEN_ZERO_OPTIMUM_TOLERANCE.get((paper_id, instance))
    if zero_tol is not None:
        quality_ok = bool(
            is_feasible and candidate_objective is not None and abs(candidate_objective) <= zero_tol
        )
    else:
        quality_ok = bool(is_feasible and gap is not None and gap <= gap_threshold)

    limit = reference_time_limit if reference_time_limit is not None else T_MAX_LARGE
    if reference_runtime is not None and reference_runtime <= limit:
        gurobi_effective = reference_runtime
    else:
        gurobi_effective = limit
    candidate_effective = (
        min(candidate_time, candidate_time_cap) if candidate_time is not None else None
    )
    tolerance = max(tolerance_fraction * gurobi_effective, tolerance_seconds)
    fast_enough = bool(
        candidate_effective is not None
        and candidate_effective > 0
        and gurobi_effective > 0
        and candidate_effective <= gurobi_effective + tolerance
    )
    passed = quality_ok and fast_enough
    return passed, {
        "quality_ok": quality_ok,
        "fast_enough": fast_enough,
        "gap": gap,
        "gap_threshold": gap_threshold,
        "proven_zero_optimum_tolerance": zero_tol,
        "candidate_effective_time_seconds": candidate_effective,
        "reference_effective_time_seconds": gurobi_effective,
        "time_tolerance_seconds": tolerance,
    }


# ---------------------------------------------------------------------------
# upstream (FrontierOR checkout) integration
# ---------------------------------------------------------------------------

def _load_upstream(frontieror_root: Optional[str]) -> Optional[Dict[str, Any]]:
    """Import ``StagedQteScorer``/``ScoreContext`` and the contract version
    from a FrontierOR checkout (not pip-installable). ``frontieror_root`` is
    prepended to ``sys.path``; otherwise whatever is already importable
    (e.g. via PYTHONPATH) is used. Returns None if not importable."""
    if frontieror_root:
        root = os.path.abspath(os.path.expanduser(frontieror_root))
        if not os.path.isdir(os.path.join(root, "test_time_self_evolution", "scoring")):
            raise ValueError(f"frontieror_root {root!r} is not a FrontierOR checkout")
        if root not in sys.path:
            sys.path.insert(0, root)
    try:
        from test_time_self_evolution.scoring.base import ScoreContext  # type: ignore
        from test_time_self_evolution.scoring.staged_qte import StagedQteScorer  # type: ignore
    except ImportError:
        return None
    upstream: Dict[str, Any] = {"StagedQteScorer": StagedQteScorer, "ScoreContext": ScoreContext}
    try:
        from trusted_eval_infra.contracts import SCORING_CONTRACT_VERSION as upstream_version  # type: ignore
    except ImportError:
        upstream_version = None
    if upstream_version is not None and upstream_version != SCORING_CONTRACT_VERSION:
        raise ValueError(
            f"FrontierOR checkout declares scoring contract {upstream_version!r}, "
            f"but this adapter implements {SCORING_CONTRACT_VERSION!r}; refusing "
            "to label scores with the wrong contract version"
        )
    upstream["contract_version"] = upstream_version
    return upstream


# ---------------------------------------------------------------------------
# loaders for FrontierOR's own files (stdlib only)
# ---------------------------------------------------------------------------

def load_references(path: str) -> List[Dict[str, Any]]:
    """Read the dataset's public Gurobi reference table:
    ``metadata/gurobi_references.csv.gz`` (stdlib) or ``.parquet`` (needs
    ``pyarrow``). Columns: task_id, instance, objective_value, runtime,
    feasible, status, time_limit, solution_path."""
    if path.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional
            raise RuntimeError("pyarrow is required for .parquet; use gurobi_references.csv.gz") from exc
        return list(pq.read_table(path).to_pylist())
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:  # type: ignore[operator]
        return list(csv.DictReader(handle))


def load_paper_meta(path: str) -> List[Dict[str, Any]]:
    """Read ``metadata/paper_meta_info.json`` (a list of per-paper dicts
    carrying ``paper_id`` and ``direction``)."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_results_csv(path: str) -> List[Dict[str, str]]:
    """Read a one-shot results CSV (``one_shot_eval.RESULTS_CSV_COLUMNS``)."""
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_by_model(rows: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    """Group result rows by their ``model`` column; each group becomes one
    ResultSet (EvalPort test case ids must be unique within a ResultSet)."""
    out: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get("model") or ""), []).append(row)
    return out


def augmented_results_to_rows(
    results: Mapping[str, Optional[Mapping[str, Any]]],
    *,
    paper_id: str,
    model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Flatten ``eval_modes.augment_results_with_staged_qte()`` output
    (``{instance: result_dict}`` for one paper) into rows for
    ``results_to_openeval``. ``None`` entries (instances the pipeline skipped)
    are dropped; pass ``declared_instances`` to count them as missing."""
    rows: List[Dict[str, Any]] = []
    for instance, result in results.items():
        if result is None:
            continue
        row = dict(result)
        row["paper_id"] = paper_id
        row["instance"] = instance
        if model is not None:
            row.setdefault("model", model)
        rows.append(row)
    return rows


def _instance_file(instance: str) -> Optional[str]:
    if instance == "tiny":
        return "instance/tiny_instance.json"
    if instance.startswith("large_") and instance[len("large_"):].isdigit():
        return f"instance/large_instance_{instance[len('large_'):]}.json"
    return None


_PATH_FIELDS = (
    "instance_path",
    "instance_schema_path",
    "solution_schema_path",
    "mathematical_formulation_path",
    "mathematical_formulation",
)


def load_dataset_instances(
    data_dir: str,
    paper_ids: Sequence[str],
    instances: Sequence[str] = PUBLIC_INSTANCE_NAMES,
    include_formulation: bool = False,
) -> List[Dict[str, Any]]:
    """Read public task text from a local copy of the Hugging Face dataset
    (``tasks/<paper_id>/problem_description.txt``) and list each instance
    whose JSON file exists. Instance JSON is not read: it can be many
    megabytes, so only its dataset-relative path is kept. The formulation
    (``mathematical_formulation.md``) is referenced by path too, and its text
    is included only with ``include_formulation=True``."""
    out: List[Dict[str, Any]] = []
    for paper_id in paper_ids:
        task_dir = os.path.join(data_dir, "tasks", paper_id)
        desc_path = os.path.join(task_dir, "problem_description.txt")
        if not os.path.isfile(desc_path):
            raise FileNotFoundError(f"no problem_description.txt for {paper_id!r} under {task_dir}")
        with open(desc_path, encoding="utf-8") as handle:
            description = handle.read()
        formulation = None
        form_path = os.path.join(task_dir, "mathematical_formulation.md")
        has_formulation = os.path.isfile(form_path)
        if include_formulation and has_formulation:
            with open(form_path, encoding="utf-8") as handle:
                formulation = handle.read()
        for instance in instances:
            rel = _instance_file(instance)
            if rel is None or not os.path.isfile(os.path.join(task_dir, rel)):
                continue
            item: Dict[str, Any] = {
                "paper_id": paper_id,
                "instance": instance,
                "problem_description": description,
                "instance_path": f"tasks/{paper_id}/{rel}",
                "instance_schema_path": f"tasks/{paper_id}/instance_schema.json",
                "solution_schema_path": f"tasks/{paper_id}/solution_schema.json",
            }
            if has_formulation:
                item["mathematical_formulation_path"] = f"tasks/{paper_id}/mathematical_formulation.md"
            if formulation is not None:
                item["mathematical_formulation"] = formulation
            out.append(item)
    return out


def _directions(paper_meta: Any) -> Dict[str, str]:
    if paper_meta is None:
        return {}
    if isinstance(paper_meta, Mapping):
        out = {}
        for key, value in paper_meta.items():
            direction = value.get("direction") if isinstance(value, Mapping) else value
            if direction in ("min", "max"):
                out[str(key)] = direction
        return out
    return {
        str(p["paper_id"]): p["direction"]
        for p in paper_meta
        if isinstance(p, Mapping) and p.get("direction") in ("min", "max")
    }


def _paper_info(paper_meta: Any) -> Dict[str, Dict[str, Any]]:
    if paper_meta is None or isinstance(paper_meta, Mapping):
        return {}
    return {str(p["paper_id"]): dict(p) for p in paper_meta if isinstance(p, Mapping) and "paper_id" in p}


def _index_references(references: Any) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if references is None:
        raise ValueError(
            "references is required: pass the dataset's public reference table "
            "(load_references('.../metadata/gurobi_references.csv.gz')). It is both "
            "the reference source and the allow-list of public instances."
        )
    rows: Iterable[Any]
    if isinstance(references, Mapping):
        rows = []
        for key, value in references.items():
            paper_id, instance = key
            rows.append(dict(value, task_id=paper_id, instance=instance))  # type: ignore[union-attr]
    else:
        rows = references
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        paper_id = row.get("task_id", row.get("paper_id"))
        instance = row.get("instance")
        if paper_id is None or instance is None:
            continue
        key = (str(paper_id), str(instance))
        if key in index:
            raise ValueError(f"duplicate reference row for {key}")
        index[key] = {
            "objective": _to_float(row.get("objective_value")),
            "runtime": _to_float(row.get("runtime")),
            "time_limit": _to_float(row.get("time_limit")),
            "status": _to_str(row.get("status")),
            "feasible": _to_bool(row.get("feasible")),
        }
    return index


# ---------------------------------------------------------------------------
# row normalization
# ---------------------------------------------------------------------------

_PRECOMPUTED_KEYS = ("score", "stage_id", "quality_part", "speed_part", "signed_gap", "beat_amount")


def _normalize_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    paper_id = _to_str(row.get("paper_id"))
    instance = _to_str(row.get("instance"))
    if not paper_id or not instance:
        raise ValueError(f"result row needs paper_id and instance: {dict(row)!r}")
    augmented = "llm_obj" in row or "solve_time" in row
    norm: Dict[str, Any] = {
        "paper_id": paper_id,
        "instance": instance,
        "model": _to_str(row.get("model")),
        "source": "eval_modes_result" if augmented else "one_shot_csv",
        "feasible": _to_bool(row.get("feasible")),
        "candidate_objective": _to_float(row.get("llm_obj") if augmented else row.get("obj")),
        "candidate_time": _to_float(row.get("solve_time") if augmented else row.get("time")),
        "candidate_time_limit": _to_float(row.get("candidate_time_limit")),
        "status": _to_str(row.get("status")),
        "fail_reason": _to_str(row.get("fail_reason")),
        "error": _to_str(row.get("error")),
        "csv_gap": None if augmented else _to_float(row.get("gap")),
        "row_reference_objective": _to_float(row.get("gurobi_obj")),
        "direction": row.get("direction") if row.get("direction") in ("min", "max") else None,
        "precomputed": None,
    }
    if all(k in row and _to_float(row.get(k)) is not None for k in _PRECOMPUTED_KEYS):
        norm["precomputed"] = {
            "score": _to_float(row["score"]),
            "stage_id": int(_to_float(row["stage_id"]) or 0),
            "quality_part": _to_float(row["quality_part"]),
            "speed_part": _to_float(row["speed_part"]),
            "signed_gap": _to_float(row["signed_gap"]),
            "beat_amount": _to_float(row["beat_amount"]),
            "matched": bool(_to_float(row.get("matched_flag")) or 0.0),
            "beat_gurobi": bool(_to_float(row.get("beat_gurobi_flag")) or 0.0),
        }
    return norm


def _time_limit_for(
    instance: str,
    row_limit: Optional[float],
    candidate_time_limits: Union[None, float, int, Mapping[str, float]],
    reference_limit: Optional[float],
) -> Tuple[Optional[float], str]:
    if isinstance(candidate_time_limits, Mapping):
        if instance in candidate_time_limits:
            return float(candidate_time_limits[instance]), "candidate_time_limits argument"
    elif candidate_time_limits is not None:
        return float(candidate_time_limits), "candidate_time_limits argument"
    if row_limit is not None:
        return row_limit, "result row candidate_time_limit"
    if reference_limit is not None:
        return reference_limit, "reference time_limit (declared budget)"
    return None, "unknown"


def _quality_reason(norm: Dict[str, Any], ref_obj: Optional[float]) -> str:
    if norm.get("missing"):
        return "undefined: no result for this declared instance (contract scores it 0.0)"
    if norm["status"] == "gate_fail":
        return "undefined: not run (tiny-instance gate failed)"
    if norm["feasible"] is False:
        return "undefined: solution not feasible, so its objective is not a valid objective"
    if norm["feasible"] is not True:
        detail = norm["fail_reason"] or norm["status"] or "no feasibility verdict"
        if norm["error"]:
            detail += ": " + norm["error"][:80]
        return f"undefined: no checked solution ({detail})"
    if norm["candidate_objective"] is None:
        return "undefined: no candidate objective"
    if ref_obj is None:
        return "undefined: no reference objective"
    return "undefined"


# ---------------------------------------------------------------------------
# public conversions
# ---------------------------------------------------------------------------

def results_to_openeval(
    instance_results: Sequence[Mapping[str, Any]],
    *,
    references: Any,
    run_id: str,
    started_at: str,
    paper_meta: Any = None,
    suite_id: str = "frontieror",
    completed_at: Optional[str] = None,
    model: Optional[str] = None,
    declared_instances: Optional[Mapping[str, Sequence[str]]] = None,
    candidate_time_limits: Union[None, float, int, Mapping[str, float]] = None,
    stage_boundary: float = STAGE_BOUNDARY,
    staged_qte_source: str = "auto",
    frontieror_root: Optional[str] = None,
    on_non_public: str = "error",
    runner: Optional[Mapping[str, str]] = None,
    group: Optional[Mapping[str, Any]] = None,
    dataset_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert FrontierOR per-instance results for ONE model into an EvalPort
    ``ResultSet``: one ``Result`` per (paper_id, instance), id
    ``"<paper_id>:<instance>"``.

    ``instance_results`` rows can be either shape FrontierOR produces:

    - one-shot CSV rows (``read_results_csv``; columns
      ``one_shot_eval.RESULTS_CSV_COLUMNS``: ``obj``, ``time``, ``gap``,
      ``feasible``, ``status``, ...), or
    - self-evolution results from ``augment_results_with_staged_qte()``
      (flatten with ``augmented_results_to_rows``; ``llm_obj``,
      ``solve_time``, ``score``, ``stage_id``, ...).

    ``references``: the public ``gurobi_references`` table (list of rows or
    ``{(task_id, instance): row}``). Required. ``paper_meta``:
    ``paper_meta_info.json`` content, for each paper's optimization
    direction (no default direction is ever assumed, as in FrontierOR).

    ``staged_qte_source``: ``"auto"`` uses a precomputed score when the row
    has one, else FrontierOR's own ``StagedQteScorer`` if a checkout is
    importable (``frontieror_root`` or PYTHONPATH), else the built-in port.
    ``"precomputed"``, ``"upstream"`` and ``"builtin"`` force one source.

    ``declared_instances``: ``{paper_id: [instance, ...]}``. Declared
    instances with no row get a synthesized "missing" Result, so the
    contract aggregate (missing = 0.0) can be computed faithfully.

    ``candidate_time_limits``: the candidate's declared budget per instance
    (a number, or ``{instance: seconds}``), used by staged_qte's speed term.
    Defaults to the row's ``candidate_time_limit``, then the reference
    ``time_limit``.
    """
    if not instance_results and not declared_instances:
        raise ValueError("instance_results is empty -- nothing to convert")
    if staged_qte_source not in ("auto", "precomputed", "upstream", "builtin"):
        raise ValueError(f"unknown staged_qte_source {staged_qte_source!r}")
    if on_non_public not in ("error", "omit"):
        raise ValueError("on_non_public must be 'error' or 'omit'")
    if not 0.0 <= float(stage_boundary) < 1.0:
        raise ValueError("stage_boundary must be in [0, 1)")
    stage_boundary = float(stage_boundary)

    ref_index = _index_references(references)
    directions = _directions(paper_meta)
    upstream = None
    if staged_qte_source in ("auto", "upstream"):
        upstream = _load_upstream(frontieror_root)
        if upstream is None and staged_qte_source == "upstream":
            raise ImportError(
                "staged_qte_source='upstream' but FrontierOR's StagedQteScorer is not "
                "importable; pass frontieror_root=<path to a FrontierOR checkout>"
            )

    rows = [_normalize_row(r) for r in instance_results]
    models = {r["model"] for r in rows if r["model"]}
    if model is not None:
        models.add(model)
    if len(models) > 1:
        raise ValueError(
            f"rows contain several models {sorted(models)}; convert one model per "
            "ResultSet (see split_by_model)"
        )
    model_name = next(iter(models)) if models else None

    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for norm in rows:
        key = (norm["paper_id"], norm["instance"])
        if key in seen:
            raise ValueError(f"duplicate result row for {key}")
        seen[key] = norm
    if declared_instances:
        for paper_id, instances in declared_instances.items():
            for instance in instances:
                key = (str(paper_id), str(instance))
                if key not in seen:
                    seen[key] = {
                        "paper_id": key[0], "instance": key[1], "model": model_name,
                        "source": "declared_missing", "missing": True, "feasible": None,
                        "candidate_objective": None, "candidate_time": None,
                        "candidate_time_limit": None, "status": None, "fail_reason": None,
                        "error": None, "csv_gap": None, "row_reference_objective": None,
                        "direction": None, "precomputed": None,
                    }

    # Visibility gate: nothing leaves unless the instance is public.
    non_public = [k for k in seen if k not in ref_index]
    if non_public:
        if on_non_public == "error":
            raise NonPublicInstanceError(
                f"{len(non_public)} result(s) are for instances not in the public "
                "reference table; FrontierOR's visibility_contract() makes their "
                "reference data and per-instance scores TRUSTED_ONLY. Pass "
                "on_non_public='omit' to drop them (only the count is recorded)."
            )
        for key in non_public:
            del seen[key]
    if not seen:
        raise ValueError("no public results left to convert")

    # Tiny gate (one-shot protocol): large cells of a paper whose tiny row
    # did not pass are excluded from the paper's binary QTE.
    tiny_status = {
        k[0]: v["status"] for k, v in seen.items() if k[1] == "tiny" and v.get("status") is not None
    }

    results: List[Dict[str, Any]] = []
    scorer_sources: Dict[str, int] = {}
    staged_raw_all: List[float] = []
    quality_scores: List[Optional[float]] = []
    grid_cells = 0
    grid_passed = 0
    stage_counts = {0: 0, 1: 0, 2: 0}

    for key in sorted(seen, key=lambda k: (k[0], PUBLIC_INSTANCE_NAMES.index(k[1]) if k[1] in PUBLIC_INSTANCE_NAMES else 99, k[1])):
        norm = seen[key]
        paper_id, instance = key
        ref = ref_index[key]
        direction = norm.get("direction") or directions.get(paper_id)
        if direction is None:
            raise ValueError(
                f"no optimization direction for paper {paper_id!r}: pass paper_meta "
                "(metadata/paper_meta_info.json); no default is assumed"
            )
        ref_obj = ref["objective"] if ref["feasible"] is not False else None
        row_ref = norm.get("row_reference_objective")
        if row_ref is not None and ref_obj is not None and not math.isclose(row_ref, ref_obj, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                f"{key}: result row's gurobi_obj {row_ref!r} differs from the public "
                f"reference {ref_obj!r}; the row was scored against a different baseline"
            )
        time_limit, time_limit_source = _time_limit_for(
            instance, norm.get("candidate_time_limit"), candidate_time_limits, ref["time_limit"]
        )
        missing = bool(norm.get("missing"))
        c = norm["candidate_objective"]
        feasible = norm["feasible"]

        # --- staged_qte (FrontierOR's score, unchanged) ---
        if missing:
            staged_score, staged_debug, source = 0.0, _empty_debug("missing_result"), "contract (missing instance = 0.0)"
        elif norm["precomputed"] is not None and staged_qte_source in ("auto", "precomputed"):
            pre = norm["precomputed"]
            staged_score = round(float(pre["score"]), 6)
            staged_debug = {k: pre[k] for k in ("stage_id", "quality_part", "speed_part", "signed_gap", "beat_amount", "matched", "beat_gurobi")}
            if staged_debug["stage_id"] == 0:
                staged_debug["reason"] = "infeasible_or_missing (precomputed; FrontierOR does not store the reason)"
            source = "precomputed (augment_results_with_staged_qte)"
        elif staged_qte_source == "precomputed":
            raise ValueError(f"{key}: staged_qte_source='precomputed' but the row has no precomputed score")
        else:
            if time_limit is None:
                raise ValueError(f"{key}: no candidate time limit; pass candidate_time_limits")
            scorer_input = {"feasible": feasible, "llm_obj": c, "solve_time": norm["candidate_time"]}
            if upstream is not None:
                ctx = upstream["ScoreContext"](
                    time_limit=time_limit,
                    gurobi_time=ref["runtime"],
                    gurobi_time_limit=ref["time_limit"],
                    gurobi_obj=ref_obj,
                    direction=direction,
                    paper_id=paper_id,
                    instance=instance,
                )
                staged_score, staged_debug = upstream["StagedQteScorer"](stage_boundary=stage_boundary).score_instance(scorer_input, ctx)
                staged_debug = dict(staged_debug)
                source = "upstream StagedQteScorer"
            else:
                staged_score, staged_debug = staged_qte_builtin(
                    scorer_input,
                    time_limit=time_limit,
                    gurobi_obj=ref_obj,
                    gurobi_time=ref["runtime"],
                    gurobi_time_limit=ref["time_limit"],
                    direction=direction,
                    stage_boundary=stage_boundary,
                )
                source = "adapter builtin port of StagedQteScorer (staged-qte-v1)"
        scorer_sources[source] = scorer_sources.get(source, 0) + 1
        staged_raw_all.append(float(staged_score))
        stage_id = int(staged_debug.get("stage_id", 0))
        stage_counts[stage_id] = stage_counts.get(stage_id, 0) + 1

        # --- quality-only (normalized, NOT staged_qte) ---
        g: Optional[float] = None
        if not missing and feasible is True and ref_obj is not None:
            if c is not None:
                g = signed_gap(c, ref_obj, direction)
            elif norm["precomputed"] is not None and stage_id != 0:
                g = norm["precomputed"]["signed_gap"]
        if g is not None:
            q_score: Optional[float] = quality_only_score(g)
            q_passed = max(0.0, g) <= stage_boundary
            q_reason = None
        else:
            q_score, q_passed = None, False
            q_reason = _quality_reason(norm, ref_obj)
        quality_scores.append(q_score)

        # --- binary QTE (paper's beat_gurobi_1) ---
        in_grid = instance in LARGE_INSTANCES
        gate = tiny_status.get(paper_id)
        gate_blocked = in_grid and gate is not None and gate != "pass"
        gap_for_binary = norm["csv_gap"] if norm["source"] == "one_shot_csv" else _oneshot_gap(c, ref_obj, direction)
        cap = T_MAX_LARGE if in_grid else (ref["time_limit"] or T_MAX_LARGE)
        b_passed, b_meta = binary_qte(
            paper_id=paper_id,
            instance=instance,
            feasible=None if (missing or gate_blocked) else feasible,
            candidate_objective=c,
            candidate_time=norm["candidate_time"],
            gap=gap_for_binary,
            reference_runtime=ref["runtime"],
            reference_time_limit=ref["time_limit"],
            candidate_time_cap=cap,
        )
        b_meta["in_paper_beat_gurobi_1_grid"] = in_grid
        b_meta["tiny_gate"] = "not_applicable" if not in_grid else (gate if gate is not None else "unknown")
        if gate_blocked:
            b_meta["excluded_by_tiny_gate"] = True
        if in_grid:
            grid_cells += 1
            grid_passed += int(b_passed)

        frontier_meta: Dict[str, Any] = {
            "paper_id": paper_id,
            "instance": instance,
            "model": norm.get("model") or model_name,
            "direction": direction,
            "input_source": norm["source"],
            "feasible": feasible,
            "status": norm["status"],
            "fail_reason": norm["fail_reason"],
            "candidate_objective": c,
            "candidate_time_seconds": norm["candidate_time"],
            "candidate_time_limit_seconds": time_limit,
            "candidate_time_limit_source": time_limit_source,
            "reference_objective": ref["objective"],
            "reference_runtime_seconds": ref["runtime"],
            "reference_time_limit_seconds": ref["time_limit"],
            "reference_status": ref["status"],
            "visibility": "public (instance is in the dataset's public gurobi_references table)",
        }
        if missing:
            frontier_meta["missing_result"] = True
        if norm["error"]:
            frontier_meta["error_message"] = norm["error"][:500]

        staged_meta = dict(staged_debug)
        staged_meta.update(
            {
                "staged_qte": staged_score,
                "contract_version": SCORING_CONTRACT_VERSION,
                "stage_boundary": stage_boundary,
                "scorer_source": source,
                "score_slot": "null: staged_qte is unbounded above and is not clamped or rescaled",
            }
        )
        if stage_id == 0:
            staged_meta["signed_gap_note"] = "1.0 is FrontierOR's sentinel for stage 0, not a measured gap"

        quality_meta: Dict[str, Any] = {
            "signed_gap": None if g is None else round(g, 6),
            "stage_boundary": stage_boundary,
            "is_staged_qte": False,
            "formula": "min(1, max(0, 1 - g))",
        }
        if g is not None and g < 0:
            quality_meta["clamped_from"] = round(1.0 - g, 6)

        grader_results: List[Dict[str, Any]] = [
            {
                "grader_id": GRADER_QUALITY_ONLY,
                "type": "custom",
                "score": q_score,
                "passed": bool(q_passed),
                "metadata": {_NS: quality_meta},
            },
            {
                "grader_id": GRADER_STAGED_QTE,
                "type": "custom",
                "score": None,
                # EvalPort Validation Rule 6: a null score means "not verified" and
                # MUST have passed false. The stage verdict is metadata stage_id.
                "passed": False,
                "reason": (
                    f"staged_qte={staged_score} (stage {stage_id}); not placed in the [0,1] "
                    "score slot because staged_qte is unbounded above, so passed is false "
                    "(EvalPort Rule 6); see metadata.frontieror.stage_id"
                ),
                "metadata": {_NS: staged_meta},
            },
            {
                "grader_id": GRADER_BINARY_QTE,
                "type": "custom",
                "score": 1.0 if b_passed else 0.0,
                "passed": bool(b_passed),
                "metadata": {_NS: b_meta},
            },
        ]
        if q_reason is not None:
            grader_results[0]["reason"] = q_reason

        result: Dict[str, Any] = {
            "test_case_id": f"{paper_id}:{instance}",
            "passed": bool(b_passed),
            "grader_results": grader_results,
            "metadata": {_NS: frontier_meta},
        }
        if norm["candidate_time"] is not None and norm["candidate_time"] >= 0:
            result["duration_ms"] = int(round(norm["candidate_time"] * 1000))
        error_text = (norm["error"] or "").lower()
        if norm["fail_reason"] == "timeout" or "timed out" in error_text:
            result["error"] = {"type": "timeout", "message": (norm["error"] or "timed out")[:500]}
        results.append(result)

    total = len(results)
    n_passed = sum(1 for r in results if r["passed"])
    non_null = [s for s in quality_scores if s is not None]
    q_pass = sum(1 for r in results if r["grader_results"][0]["passed"])
    summary: Dict[str, Any] = {
        "total": total,
        "passed": n_passed,
        "failed": total - n_passed,
        "pass_rate": n_passed / total,
        "by_grader": {
            GRADER_QUALITY_ONLY: {"passed": q_pass, "failed": total - q_pass},
            # Null-scored ("not verified", Rule 6), so excluded from pass/fail
            # counts. Per-stage counts: metadata.frontieror.aggregates.stage_counts.
            GRADER_STAGED_QTE: {"passed": 0, "failed": 0},
            GRADER_BINARY_QTE: {"passed": n_passed, "failed": total - n_passed, "avg_score": n_passed / total},
        },
    }
    if non_null:
        summary["avg_score"] = round(sum(non_null) / len(non_null), 6)
        summary["by_grader"][GRADER_QUALITY_ONLY]["avg_score"] = summary["avg_score"]

    aggregates: Dict[str, Any] = {
        "staged_qte_mean": round(sum(staged_raw_all) / total, 6),
        "staged_qte_mean_rule": (
            "contract: arithmetic mean over the results in this ResultSet, missing/"
            "failed instances = 0.0. Equals FrontierOR's aggregate only when every "
            "declared instance is present (declared_instances_complete)."
        ),
        "declared_instances_complete": bool(declared_instances),
        "quality_only_mean_null_as_zero": round(sum(non_null) / total, 6),
        "quality_only_mean_non_null": summary.get("avg_score"),
        "quality_only_null_count": total - len(non_null),
        "stage_counts": {str(k): v for k, v in sorted(stage_counts.items())},
        "binary_qte_large_cells": grid_cells,
        "binary_qte_large_passed": grid_passed,
        "binary_qte_large_rate": (grid_passed / grid_cells) if grid_cells else None,
        "binary_qte_rate_rule": (
            "beat_gurobi_1 over the large_* results present. The paper divides by "
            "5 x |papers| over a fixed anchor set, so this equals the paper's number "
            "only when all five large instances of every paper are present."
        ),
    }

    rs_meta: Dict[str, Any] = {
        "source": "frontieror",
        "adapter": f"frontieror-openeval-adapter {__version__}",
        "contract": {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "contract_version": SCORING_CONTRACT_VERSION,
            "scorer": "staged_qte",
            "stage_boundary": stage_boundary,
            "scale_D": "abs(r) if abs(r)>=0.001 else max(abs(c),0.001)",
            "per_instance_decimal_places": INSTANCE_SCORE_DECIMALS,
            "aggregation": "arithmetic_mean_over_declared_instances; missing_or_failed_instance_score = 0.0",
        },
        "summary_note": (
            "summary.avg_score is the EvalPort-style mean of the non-null "
            "quality_only scores, which is NOT a FrontierOR metric. FrontierOR's own "
            "aggregates are in metadata.frontieror.aggregates."
        ),
        "aggregates": aggregates,
        "staged_qte_scorer_sources": scorer_sources,
        "visibility_policy": (
            "Only instances in the public reference table are exported; "
            "reference data for hidden/trusted-only instances is never written."
        ),
        "non_public_results_omitted": len(non_public),
    }
    if upstream is not None and upstream.get("contract_version"):
        rs_meta["upstream_contract_version_checked"] = upstream["contract_version"]
    if dataset_revision:
        rs_meta["dataset_revision"] = dataset_revision

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": summary,
        "metadata": {
            _NS: rs_meta,
            # PROPOSED (evalport Discussion #49, alternative B), NOT in the spec on
            # main: a Result with some null-scored and some scored graders must say
            # how its `passed` was derived. Here `passed` is the binary QTE grader's
            # verdict alone, and the staged_qte grader's null is by design (unbounded
            # score kept in metadata), not a grader that failed to run -- so the
            # honest declaration is `producer`: consumers must not re-derive passed
            # from the graders. Harmless on main, which ignores unknown metadata.
            "openeval": {"aggregation": {"strategy": "producer"}},
        },
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    if model_name:
        result_set["provider"] = {"model": model_name}
    if runner:
        result_set["runner"] = dict(runner)
    if group:
        result_set["group"] = dict(group)
    return result_set


def to_openeval(
    dataset_instances: Sequence[Mapping[str, Any]],
    *,
    paper_meta: Any = None,
    suite_id: str = "frontieror",
    name: Optional[str] = None,
    description: Optional[str] = None,
    stage_boundary: float = STAGE_BOUNDARY,
    dataset_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert public FrontierOR task instances (``load_dataset_instances``)
    into an EvalPort ``Suite``: one ``TestCase`` per (paper_id, instance), id
    ``"<paper_id>:<instance>"``, input = the paper's public problem
    description. Instance data is referenced by dataset path, not embedded.
    No reference objective or runtime is put in the suite."""
    if not dataset_instances:
        raise ValueError("dataset_instances is empty -- nothing to convert")
    directions = _directions(paper_meta)
    info = _paper_info(paper_meta)
    graders = _graders(float(stage_boundary))
    grader_ids = [g["id"] for g in graders]
    test_cases: List[Dict[str, Any]] = []
    for item in dataset_instances:
        paper_id = _to_str(item.get("paper_id"))
        instance = _to_str(item.get("instance"))
        text = item.get("problem_description")
        if not paper_id or not instance:
            raise ValueError(f"dataset instance needs paper_id and instance: {dict(item)!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{paper_id}:{instance} has no problem_description")
        meta: Dict[str, Any] = {
            "paper_id": paper_id,
            "instance": instance,
            "visibility": "public",
        }
        for field in _PATH_FIELDS:
            if item.get(field):
                meta[field] = item[field]
        if paper_id in directions:
            meta["direction"] = directions[paper_id]
        for field in ("paper_title", "publication", "year", "problem_class", "formulation_type", "application_field", "source_link"):
            if paper_id in info and info[paper_id].get(field) is not None:
                meta[field] = info[paper_id][field]
        test_cases.append(
            {
                "id": f"{paper_id}:{instance}",
                "input": text,
                "graders": list(grader_ids),
                "metadata": {_NS: meta},
            }
        )
    suite_meta: Dict[str, Any] = {"source": "frontieror", "contract_version": SCORING_CONTRACT_VERSION}
    if dataset_revision:
        suite_meta["dataset_revision"] = dataset_revision
    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "graders": graders,
        "test_cases": test_cases,
        "metadata": {_NS: suite_meta},
    }
    if name is not None:
        suite["name"] = name
    if description is not None:
        suite["description"] = description
    return suite


def from_openeval(suite: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Reverse ``to_openeval``: recover the dataset-instance dicts. Test cases
    without this adapter's ``frontieror`` metadata namespace are skipped."""
    out: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        meta = (tc.get("metadata") or {}).get(_NS)
        if not meta:
            continue
        item: Dict[str, Any] = {
            "paper_id": meta["paper_id"],
            "instance": meta["instance"],
            "problem_description": tc.get("input"),
        }
        for field in _PATH_FIELDS:
            if field in meta:
                item[field] = meta[field]
        out.append(item)
    return out
