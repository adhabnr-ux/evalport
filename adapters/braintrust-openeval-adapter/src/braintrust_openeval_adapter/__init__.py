"""Braintrust <-> EvalPort adapter.

Standalone converter between Braintrust (https://www.braintrust.dev)
`Eval()` run results and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package rather than living inside the
Braintrust SDK itself: it follows the same playbook that already worked for
AutoGen, CrewAI, Ragas, and LangSmith (see ../autogen-openeval-adapter,
../crewai-openeval-adapter, ../ragas-openeval-adapter,
../langsmith-openeval-adapter) — it works against Braintrust's public
`Eval()` result shape (a summary exposing per-case `input`/`expected`/
`output`/`scores`) from the outside, so you get EvalPort import/export
today without needing anything merged into the `braintrust` SDK.

Tracked as https://github.com/adhabnr-ux/evalport/issues/3.
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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object.

    Braintrust's `EvalResultWithSummary`/`EvalCase` result objects and
    JSON-loaded eval output both show up in the wild, so every accessor in
    this module goes through here rather than assuming one shape.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cases_from_result(braintrust_result: Any) -> List[Any]:
    """Normalize a Braintrust Eval() result (or dict/list stand-in) into a list of per-case results.

    `Eval()` returns an `EvalResultWithSummary` exposing `.results` (a list
    of `EvalCase`-shaped entries with `input`/`expected`/`output`/`scores`).
    Also accepts a plain dict with a `results` key, or a bare list of cases
    (as used in tests / JSON-loaded output), so no direct `braintrust`
    import is required.
    """
    if isinstance(braintrust_result, list):
        return list(braintrust_result)
    results = _get(braintrust_result, "results", None)
    return list(results) if results is not None else []


def _case_payload(case: Any, index: int) -> Dict[str, Any]:
    """Normalize a single Braintrust eval case result into an EvalPort TestCase dict."""
    case_id = _get(case, "id") or f"tc_{index}"
    input_value = _get(case, "input", "")
    expected = _get(case, "expected", None)
    output = _get(case, "output", None)
    scores = dict(_get(case, "scores", None) or {})

    graders = [f"gr_{name}" for name in sorted(scores.keys())] or ["gr_braintrust_score"]

    tc: Dict[str, Any] = {
        "id": str(case_id),
        "input": input_value if isinstance(input_value, (str, list)) else str(input_value),
        "graders": graders,
    }
    if expected is not None:
        tc["expected_output"] = expected if isinstance(expected, str) else str(expected)

    metadata: Dict[str, Any] = {}
    if scores:
        metadata["braintrust_scores"] = scores
    if output is not None:
        # The case's actual output belongs on a Result, not a TestCase —
        # kept as metadata so round-tripping doesn't lose it.
        metadata["braintrust_actual_output"] = output if isinstance(output, str) else str(output)
    if metadata:
        tc["metadata"] = metadata
    return tc


def to_openeval(braintrust_result: Any, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Export a Braintrust `Eval()` result to an EvalPort-shaped suite (dict).

    `braintrust_result` may be a real Braintrust `EvalResultWithSummary`
    (uses its `.results` list), a plain dict with a `results` key, or a
    bare list of case dicts — no direct `braintrust` import is required.
    Each case is expected to expose `input`, `expected`, `output`, and
    `scores` (a dict of scorer name -> numeric score), matching what
    Braintrust's `Eval()` produces per test case.

    Every scorer present on a case (e.g. `"Factuality"`, `"ExactMatch"`,
    a custom scorer function's name) becomes its own EvalPort grader
    (`gr_<name>`, type "custom", handler `braintrust:<name>`) so a
    downstream EvalPort runner can re-score with the same scorer set. The
    scores Braintrust already computed are preserved per test case under
    `metadata.braintrust_scores` rather than discarded, since an `Eval()`
    run is already-scored data, not just a task definition.

    Returns a plain dict conforming to the EvalPort EvalSuite schema. Pass
    it to `openeval.validate.validate_suite()` to confirm compliance, or
    `json.dump()` it directly to share as a `.json` suite file.
    """
    cases = _cases_from_result(braintrust_result)
    resolved_run_id = run_id or _get(braintrust_result, "experimentName") or _get(braintrust_result, "id") or "braintrust_run"

    test_cases = [_case_payload(c, i) for i, c in enumerate(cases)]

    scorer_names = sorted(
        {name for tc in test_cases for name in (tc.get("metadata") or {}).get("braintrust_scores", {}).keys()}
    )
    graders = [
        {
            "id": f"gr_{name}",
            "type": "custom",
            "description": f"Braintrust '{name}' scorer",
            "params": {"handler": f"braintrust:{name}"},
        }
        for name in scorer_names
    ]
    if not graders:
        graders = [{"id": "gr_braintrust_score", "type": "custom", "params": {"handler": "braintrust:score"}}]

    return {
        "version": OPENEVAL_VERSION,
        "id": f"braintrust_eval_{resolved_run_id}",
        "name": f"Braintrust eval run {resolved_run_id}",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "braintrust"}, "braintrust_scorers": scorer_names},
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Import an EvalPort suite into a list of Braintrust-shaped case dicts.

    Returns plain dicts with `input`/`expected` keys, ready to pass as the
    `data` argument of Braintrust's `Eval()`:

        from braintrust import Eval
        from braintrust_openeval_adapter import from_openeval

        cases = from_openeval(suite)
        Eval("my-project", data=cases, task=my_task, scores=[...])
    """
    cases: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []):
        case: Dict[str, Any] = {"input": tc.get("input")}
        if tc.get("expected_output") is not None:
            case["expected"] = tc.get("expected_output")
        cases.append(case)
    return cases
