"""Ragas <-> EvalPort adapter.

Standalone converter between Ragas (https://github.com/explodinggradients/ragas)
evaluation results and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside Ragas
itself: it follows the same playbook that already worked for AutoGen and
CrewAI (see ../autogen-openeval-adapter and ../crewai-openeval-adapter) —
it works against Ragas's public `EvaluationResult` shape (the object
returned by `ragas.evaluate()`, exposing `.scores` and `.to_pandas()`) from
the outside, so you get EvalPort import/export today without needing
anything merged into Ragas's core.

Tracked as https://github.com/adhabnr-ux/evalport/issues/1.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "__version__"]
__version__ = "0.1.0"

# Ragas metric column names that show up in EvaluationResult.to_pandas() /
# .scores in addition to the sample's own input/output/context columns.
# Anything in this set is treated as a per-sample metric score rather than
# as one of the sample's own fields.
_KNOWN_METRIC_NAMES = {
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "context_entity_recall",
    "answer_similarity",
    "answer_correctness",
    "summarization_score",
    "harmfulness",
    "maliciousness",
    "coherence",
    "correctness",
    "conciseness",
}

# Column names (in Ragas's dataframe / sample dict) that map to standard
# EvalPort TestCase fields rather than being treated as metric scores or
# passed through as opaque metadata.
_INPUT_KEYS = ("user_input", "question")
_OUTPUT_KEYS = ("response", "answer")
_EXPECTED_KEYS = ("reference", "ground_truth")
_CONTEXT_KEYS = ("retrieved_contexts", "contexts")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object (or pandas Series)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        # pandas Series supports both `in` and `[]`
        if key in obj:  # type: ignore[operator]
            return obj[key]
        return default
    except TypeError:
        pass
    return getattr(obj, key, default)


def _samples_from_result(ragas_result: Any) -> List[Dict[str, Any]]:
    """Normalize a Ragas EvaluationResult (or dict/list stand-in) into a list of sample dicts.

    Prefers `.to_pandas()` (the documented, stable way to get per-sample rows
    out of an EvaluationResult) and falls back to `.scores` (a list of
    per-sample score dicts also exposed directly on the result) or, for
    tests and JSON-loaded output, a plain list of dicts.
    """
    to_pandas = getattr(ragas_result, "to_pandas", None)
    if callable(to_pandas):
        df = to_pandas()
        return [row.to_dict() for _, row in df.iterrows()]

    scores = _get(ragas_result, "scores", None)
    if scores is not None:
        return list(scores)

    if isinstance(ragas_result, list):
        return list(ragas_result)

    return []


def _first(sample: Dict[str, Any], keys: tuple, default: Any = None) -> Any:
    for k in keys:
        if k in sample and sample[k] is not None:
            return sample[k]
    return default


def _sample_payload(sample: Dict[str, Any], index: int) -> Dict[str, Any]:
    """Normalize one Ragas per-sample row into an EvalPort TestCase dict."""
    input_text = _first(sample, _INPUT_KEYS, "")
    output_text = _first(sample, _OUTPUT_KEYS, None)
    expected = _first(sample, _EXPECTED_KEYS, None)
    contexts = _first(sample, _CONTEXT_KEYS, None)

    metric_scores = {
        k: v
        for k, v in sample.items()
        if k in _KNOWN_METRIC_NAMES and v is not None
    }
    graders = [f"gr_{name}" for name in sorted(metric_scores.keys())] or ["gr_ragas_score"]

    tc: Dict[str, Any] = {
        "id": f"tc_{index}",
        "input": input_text if isinstance(input_text, (str, list)) else str(input_text),
        "graders": graders,
    }
    if expected is not None:
        tc["expected_output"] = str(expected)
    if contexts:
        tc["context"] = list(contexts) if not isinstance(contexts, str) else [contexts]

    metadata: Dict[str, Any] = {"ragas_scores": metric_scores} if metric_scores else {}
    if output_text is not None:
        # Ragas evaluates output that was already generated elsewhere — keep
        # it as metadata (not `actual_output`, which belongs on a Result,
        # not a TestCase) so round-tripping doesn't lose it.
        metadata["ragas_actual_output"] = str(output_text)
    if metadata:
        tc["metadata"] = metadata
    return tc


def to_openeval(ragas_result: Any, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Export a Ragas `evaluate()` result to an EvalPort-shaped suite (dict).

    `ragas_result` may be a real Ragas `EvaluationResult` (uses its
    `to_pandas()` method), a plain object/dict exposing `.scores`, or a
    plain list of per-sample dicts — no direct Ragas import is required.

    Each Ragas metric present on a sample (`faithfulness`,
    `answer_relevancy`, `context_precision`, `context_recall`, etc.)
    becomes its own EvalPort grader (`gr_<metric>`, type "custom", handler
    `ragas:<metric>`) so a downstream EvalPort runner can re-score with the
    same metric set. The scores Ragas already computed are preserved
    per-sample under `test_case.metadata.ragas_scores` rather than
    discarded, since `evaluate()` output is scored data, not just a task
    definition.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    samples = _samples_from_result(ragas_result)
    resolved_run_id = run_id or _get(ragas_result, "run_id") or _get(ragas_result, "id") or "ragas_run"

    test_cases = [_sample_payload(s, i) for i, s in enumerate(samples)]

    metric_names = sorted(
        {name for tc in test_cases for name in (tc.get("metadata") or {}).get("ragas_scores", {}).keys()}
    )
    graders = [
        {
            "id": f"gr_{name}",
            "type": "custom",
            "description": f"Ragas {name} metric",
            "params": {"handler": f"ragas:{name}"},
        }
        for name in metric_names
    ]
    if not graders:
        graders = [{"id": "gr_ragas_score", "type": "custom", "params": {"handler": "ragas:score"}}]

    return {
        "version": OPENEVAL_VERSION,
        "id": f"ragas_eval_{resolved_run_id}",
        "name": f"Ragas eval run {resolved_run_id}",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "ragas"}, "ragas_metrics": metric_names},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of Ragas-shaped sample dicts.

    Returns plain dicts with the `user_input` / `reference` / `retrieved_contexts`
    keys Ragas's `Dataset.from_list()` / `EvaluationDataset` expects, so you
    can build a fresh Ragas dataset to re-run `evaluate()` against:

        from datasets import Dataset
        from ragas_openeval_adapter import from_openeval

        samples = from_openeval(suite)
        dataset = Dataset.from_list(samples)
    """
    samples: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        sample: Dict[str, Any] = {
            "user_input": tc.get("input"),
        }
        if tc.get("expected_output") is not None:
            sample["reference"] = tc.get("expected_output")
        if tc.get("context"):
            sample["retrieved_contexts"] = list(tc.get("context"))
        samples.append(sample)
    return samples
