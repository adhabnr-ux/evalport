"""
arthur_bench_openeval_adapter

Converts Arthur Bench (``arthur-bench``) TestSuites, TestRuns, and scoring
results to/from EvalPort (https://github.com/adhabnr-ux/evalport), the open
interchange format for portable LLM evaluation test cases, graders, suites,
and results.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from arthur_bench.run.testsuite import TestSuite
from arthur_bench.run.testrun import TestRun
from arthur_bench.scoring import scorer_from_string
from arthur_bench.scoring.scorer import Scorer

__all__ = ["to_openeval", "from_openeval", "run_to_openeval"]

SPEC_VERSION = "1.0.0-rc.2"

# Real category taxonomies for Arthur Bench's built-in categorical scorers,
# confirmed by reading the installed package's source (arthur_bench.scoring.*),
# not assumed. Used to derive an honest `passed` boolean from a scorer's
# actual category name rather than guessing from the numeric score alone --
# several of these scorers have MORE than two categories (e.g. qa_correctness
# has "incorrect"/"correct"/"invalid"; summary_quality has four), so a
# score >= 0.5 threshold isn't a safe stand-in the way it is for a strictly
# binary scorer like exact_match.
_CATEGORY_PASS_NAMES: Dict[str, set] = {
    "exact_match": {"match"},
    "hallucination": {"no hallucination"},
    "python_unit_testing": {"pass"},
    "qa_correctness": {"correct"},
    "summary_quality": {"candidate", "equal"},
}

# Continuous (non-categorical) scorers where a HIGHER score means a WORSE
# result. Confirmed by reading each scorer's own docstring in the installed
# package: hedging_language's docstring states "higher values corresponding
# to higher likelihood of hedging language being present" -- hedging is the
# thing being detected, not a quality score, so higher is worse. Every other
# built-in continuous scorer this adapter checked (specificity, bertscore,
# word_count_match) is higher-is-better, matching the fallback below.
_INVERTED_CONTINUOUS_SCORERS = {"hedging_language"}


def _grader_for_scorer(scorer: Scorer) -> Dict[str, Any]:
    """Build an EvalPort grader dict for one Arthur Bench ``Scorer`` instance.

    Every scorer -- including ``exact_match`` -- maps to EvalPort's "custom"
    grader type. This is deliberate, not a shortcut, and for exact_match
    specifically it is NOT the obvious choice: EvalPort has its own
    `exact_match` grader type, and Arthur Bench's ExactMatch scorer really is
    a literal string-equality check. But reading the installed package's
    source (arthur_bench/scoring/exact_match.py) turned up a real, confirmed
    quirk: `ExactMatch(case_sensitive=True)` -- the default used when
    `scoring_method="exact_match"` is passed as a string -- actually performs
    a case-INSENSITIVE comparison (both sides are lowercased before
    comparing); `case_sensitive=False` is what does a true case-sensitive
    compare. That's inverted from what the parameter name implies. If this
    were mapped to EvalPort's own `exact_match` grader type and a suite
    exported here were later re-run by a different, spec-conformant
    exact_match implementation (case-sensitive by the ordinary meaning of the
    word), case-differing outputs could silently score differently than they
    did in Arthur Bench. "custom", with the scorer's real config captured via
    its own `to_dict()`, avoids that risk -- and generalizes cleanly to every
    other scorer (some of which need a live Arthur-hosted API or an LLM judge
    this adapter has no honest way to fabricate credentials for).
    """
    name = scorer.name()
    config = scorer.to_dict(warn=False)
    params: Dict[str, Any] = {"handler": name}
    if config:
        params["config"] = config

    if name == "exact_match":
        description = (
            "arthur-bench ExactMatch scorer -- literal string comparison. Note: "
            "the default case_sensitive=True config actually lowercases both "
            "sides before comparing (case-INsensitive); case_sensitive=False is "
            "the true case-sensitive compare. Real quirk in the installed "
            "package, not a mapping error here."
        )
    elif name in _CATEGORY_PASS_NAMES or name in ("bertscore", "specificity", "word_count_match"):
        description = f"arthur-bench {name} scorer (local, no live API required)"
    else:
        description = f"arthur-bench {name} scorer (may require a live Arthur-hosted API or LLM judge to run)"

    return {
        "id": name,
        "type": "custom",
        "params": params,
        "description": description,
    }


def to_openeval(
    suite: TestSuite,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert an Arthur Bench ``TestSuite`` (its test cases and scoring
    method) into a spec-valid EvalPort Suite.

    Args:
        suite: An already-constructed ``arthur_bench.run.testsuite.TestSuite``.
        suite_id: Suite id. Defaults to the TestSuite's own name.
        name: Suite name. Defaults to the TestSuite's own name.

    Returns:
        A dict matching ``spec/schemas/suite.json``.
    """
    grader = _grader_for_scorer(suite.scorer)
    grader_id = grader["id"]

    test_cases: List[Dict[str, Any]] = []
    for tc in suite.test_cases:
        test_case: Dict[str, Any] = {
            "id": str(tc.id),
            "input": tc.input,
            "graders": [grader_id],
            "metadata": {
                "arthur_bench": {"input": tc.input, "reference_output": tc.reference_output}
            },
        }
        if tc.reference_output is not None:
            test_case["expected_output"] = tc.reference_output
        test_cases.append(test_case)

    suite_dict: Dict[str, Any] = {
        "version": SPEC_VERSION,
        "id": suite_id or suite.name or f"arthur-bench-{uuid.uuid4()}",
        "test_cases": test_cases,
        "graders": [grader],
    }
    suite_dict["name"] = name or suite.name
    if suite.description:
        suite_dict["metadata"] = {"arthur_bench": {"description": suite.description}}
    return suite_dict


def from_openeval(
    suite: Dict[str, Any],
    scorer: Optional[Scorer] = None,
    client: Optional[Any] = None,
    test_suite_name: Optional[str] = None,
) -> TestSuite:
    """Convert an EvalPort Suite back into an Arthur Bench ``TestSuite``.

    Args:
        suite: A spec-valid EvalPort Suite dict with exactly one grader
            (Arthur Bench TestSuites have a single scoring method).
        scorer: Required if the suite's grader is a custom (non-built-in)
            scorer -- Arthur Bench itself refuses to reconstruct a custom
            scorer from a string name (see ``UserValueError`` in
            ``TestSuite.__init__``), so this adapter can't either. Optional
            for built-in scorers (``exact_match``, ``bertscore``, etc.),
            where it's resolved automatically from the grader's
            ``params.handler``/``params.config`` via
            ``arthur_bench.scoring.scorer_from_string`` and ``Scorer.from_dict``.
        client: Optional ``BenchClient`` to pass through to the new
            ``TestSuite`` (e.g. a ``LocalBenchClient`` pointed at a specific
            directory, for test isolation). Defaults to Arthur Bench's own
            default client resolution.
        test_suite_name: Name for the reconstructed TestSuite. Defaults to
            the EvalPort suite's own ``name``/``id``.

    Returns:
        A new ``arthur_bench.run.testsuite.TestSuite``.
    """
    graders = suite.get("graders") or []
    if len(graders) != 1:
        raise ValueError(
            "from_openeval() expects exactly one grader -- Arthur Bench TestSuites "
            f"have a single scoring method, this suite has {len(graders)}"
        )
    grader = graders[0]
    handler = grader.get("params", {}).get("handler", grader["id"])
    config = grader.get("params", {}).get("config", {})

    if scorer is None:
        try:
            scorer_cls = scorer_from_string(handler)
        except Exception as exc:
            raise ValueError(
                f"'{handler}' is not a built-in Arthur Bench scorer name -- pass the "
                "original scorer instance explicitly via from_openeval(suite, scorer=...)"
            ) from exc
        scorer = scorer_cls.from_dict(config) if config else scorer_cls()

    inputs: List[str] = []
    references: List[Optional[str]] = []
    for tc in suite.get("test_cases", []):
        metadata = tc.get("metadata") or {}
        arthur_meta = metadata.get("arthur_bench") if isinstance(metadata, dict) else None
        if isinstance(arthur_meta, dict) and "input" in arthur_meta:
            inputs.append(arthur_meta["input"])
            references.append(arthur_meta.get("reference_output"))
        else:
            input_value = tc.get("input")
            inputs.append(" ".join(input_value) if isinstance(input_value, list) else input_value)
            references.append(tc.get("expected_output"))

    reference_output_list = references if any(r is not None for r in references) else None

    kwargs: Dict[str, Any] = {
        "name": test_suite_name or suite.get("name") or suite.get("id") or f"arthur-bench-{uuid.uuid4()}",
        "scoring_method": scorer,
        "input_text_list": inputs,
        "reference_output_list": reference_output_list,
    }
    suite_metadata = suite.get("metadata") or {}
    arthur_suite_meta = suite_metadata.get("arthur_bench") if isinstance(suite_metadata, dict) else None
    if isinstance(arthur_suite_meta, dict) and arthur_suite_meta.get("description"):
        kwargs["description"] = arthur_suite_meta["description"]
    if client is not None:
        kwargs["client"] = client

    return TestSuite(**kwargs)


def _passed_for(scorer_name: str, score: Optional[float], category_name: Optional[str]) -> bool:
    if category_name is not None and scorer_name in _CATEGORY_PASS_NAMES:
        return category_name in _CATEGORY_PASS_NAMES[scorer_name]
    if score is None:
        return False
    if scorer_name in _INVERTED_CONTINUOUS_SCORERS:
        return score < 0.5
    return score >= 0.5


def run_to_openeval(
    run: TestRun,
    suite: Optional[TestSuite] = None,
    scorer_name: Optional[str] = None,
    suite_id: Optional[str] = None,
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a completed Arthur Bench ``TestRun`` into a spec-valid
    EvalPort ResultSet.

    Args:
        run: The ``TestRun`` returned by ``TestSuite.run(...)``.
        suite: The ``TestSuite`` that produced ``run`` (typical usage is
            ``suite = TestSuite(...); run = suite.run(...)``). Used to
            recover the scorer's real name -- ``TestRun``/``TestCaseOutput``
            don't carry the scorer identity themselves, only bare scores and
            categories, so without this (or ``scorer_name``) this adapter has
            no honest way to know which scorer produced a result and would
            have to fall back to a placeholder grader id.
        scorer_name: Alternative to ``suite`` -- the scorer's ``name()``
            directly (e.g. ``"exact_match"``), if the original ``TestSuite``
            object isn't available. One of ``suite``/``scorer_name`` is
            required for the grader id to be meaningful and for the
            category-aware ``passed`` logic below to apply; without either,
            results are still emitted (nothing is dropped) but under a
            generic ``"scorer"`` grader id and a numeric-only pass heuristic.
        suite_id: Id of the EvalPort suite this run corresponds to. Defaults
            to the run's own ``test_suite_id`` if set.
        run_id: Id for this run. Defaults to the run's own ``id``, or a
            generated uuid4 if that's unset (e.g. when ``run(..., save=False)``).
        started_at: ISO 8601 timestamp. Defaults to now (UTC) -- TestRun
            doesn't carry its own start timestamp.

    Returns:
        A dict matching ``spec/schemas/resultset.json``.
    """
    if scorer_name is None and suite is not None:
        scorer_name = suite.scorer.name()
    grader_id = scorer_name or "scorer"

    results: List[Dict[str, Any]] = []
    for tc in run.test_cases:
        score = tc.score
        category = tc.score_result.category if tc.score_result is not None else None
        category_name = category.name if category is not None else None

        clamped_score = None
        if score is not None:
            clamped_score = max(0.0, min(1.0, float(score)))

        passed = _passed_for(grader_id, score, category_name)

        grader_result: Dict[str, Any] = {
            "grader_id": grader_id,
            "type": "custom",
            "score": clamped_score,
            "passed": passed,
            "metadata": {
                "arthur_bench": {
                    "raw_score": score,
                    "category": category_name,
                    "category_description": category.description if category is not None else None,
                }
            },
        }
        if category is not None:
            grader_result["reason"] = f"{category.name}: {category.description}"

        results.append(
            {
                "test_case_id": str(tc.id),
                "grader_results": [grader_result],
                "passed": passed,
                "actual_output": tc.output,
            }
        )

    result_set: Dict[str, Any] = {
        "version": SPEC_VERSION,
        "suite_id": suite_id or (str(run.test_suite_id) if run.test_suite_id else "arthur-bench-run"),
        "run_id": run_id or (str(run.id) if run.id else str(uuid.uuid4())),
        "started_at": started_at or datetime.now(timezone.utc).isoformat(),
        "results": results,
        "runner": {"name": "arthur-bench-openeval-adapter", "version": "0.1.0"},
    }
    if results:
        passed_count = sum(1 for r in results if r["passed"])
        result_set["summary"] = {
            "total": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "pass_rate": passed_count / len(results),
        }
    extra_meta: Dict[str, Any] = {}
    if run.model_name:
        extra_meta["model_name"] = run.model_name
    if run.foundation_model:
        extra_meta["foundation_model"] = run.foundation_model
    if run.prompt_template:
        extra_meta["prompt_template"] = run.prompt_template
    if run.model_version:
        extra_meta["model_version"] = run.model_version
    if extra_meta:
        result_set["metadata"] = {"arthur_bench": extra_meta}
    return result_set
