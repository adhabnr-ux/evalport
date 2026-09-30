"""Export an EvalPort Suite + ResultSet as one Every Eval Ever (EEE) record.

Every Eval Ever (https://github.com/evaleval/every_eval_ever) stores *aggregate*
evaluation results per model, described by ``eval.schema.json`` (v0.3.0). This
script converts one EvalPort ``ResultSet`` (plus its ``Suite`` and some
caller-supplied context that EvalPort does not carry) into one EEE aggregate
record with ``source_metadata.source_type == "evaluation_run"``.

It is ONE-WAY, LOSSY and stdlib-only. The losses are listed in the README next
to this file. The two that matter most:

* EvalPort's ``GraderResult.score`` may be ``null`` ("not verified", SPEC
  Validation Rule 6). EEE's ``score_details.score`` is a required number and has
  no null. Null scores are therefore *excluded* from every mean, never counted
  as 0 and never counted as failures; how many were excluded is recorded as
  strings in ``score_details.details``. A grader with no non-null score at all
  gets no evaluation result (and is named in
  ``source_metadata.additional_details``).
* Per-test-case results (inputs, outputs, per-case scores) are not exported;
  EEE carries those in a separate instance-level JSONL that this script does
  not write.

Nothing EEE requires but EvalPort lacks is invented: the model id, deployment
type, weight availability, source organization and evaluator relationship must
be supplied by the caller, and the conversion fails with a clear message when
they are missing.

Usage::

    python evalport_to_eee.py SUITE.json RESULTSET.json \\
        --model-id openai/gpt-4o --deployment-type externally_managed \\
        --model-availability closed_weights \\
        --source-organization-name "Example Org" \\
        --evaluator-relationship third_party > record.json

Only the standard library is needed to convert. Validating the record against
the vendored EEE schema needs ``jsonschema`` (see the tests next to this file).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

# Schema version the record is written against (vendor/eval.schema.json "version").
EEE_SCHEMA_VERSION = "0.3.0"

# Enums copied from vendor/eval.schema.json (and the EEE validator's semantic
# checks for the two model_info.additional_details axes). The tests compare these
# against the vendored schema so they cannot drift silently.
DEPLOYMENT_TYPES = ("self_deployed", "externally_managed", "unknown")
MODEL_AVAILABILITIES = ("open_weights", "closed_weights", "unknown")
EVALUATOR_RELATIONSHIPS = ("first_party", "third_party", "collaborative", "other")

STANDARD_ERROR_METHOD = "analytic"


class ConversionError(ValueError):
    """Raised when the inputs cannot be converted without inventing values."""


@dataclass
class EEEContext:
    """Facts EEE requires (or accepts) that an EvalPort ResultSet does not carry.

    The first five fields are required; there are deliberately no defaults for
    them, because guessing a model identity or who ran the evaluation would put
    wrong data in a shared datastore. Pass ``"unknown"`` explicitly for
    ``deployment_type`` / ``model_availability`` if you genuinely do not know.
    """

    model_id: str
    deployment_type: str
    model_availability: str
    source_organization_name: str
    evaluator_relationship: str
    model_name: Optional[str] = None
    developer: Optional[str] = None
    inference_platform: Optional[str] = None
    source_organization_url: Optional[str] = None
    eval_library_name: Optional[str] = None
    eval_library_version: Optional[str] = None
    # Where the evaluated dataset lives. At most one of dataset_urls / hf_repo.
    dataset_urls: List[str] = field(default_factory=list)
    hf_repo: Optional[str] = None
    hf_split: Optional[str] = None
    # Unix epoch string of record creation. None means "now".
    retrieved_timestamp: Optional[str] = None

    def validate(self) -> None:
        missing = [
            name
            for name in (
                "model_id",
                "deployment_type",
                "model_availability",
                "source_organization_name",
                "evaluator_relationship",
            )
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip()
        ]
        if missing:
            raise ConversionError(
                "missing required EEE context (EvalPort does not carry it, and it is not "
                "guessed): " + ", ".join(missing)
            )
        if self.deployment_type not in DEPLOYMENT_TYPES:
            raise ConversionError(
                f"deployment_type must be one of {list(DEPLOYMENT_TYPES)}, got {self.deployment_type!r}"
            )
        if self.model_availability not in MODEL_AVAILABILITIES:
            raise ConversionError(
                f"model_availability must be one of {list(MODEL_AVAILABILITIES)}, got {self.model_availability!r}"
            )
        if self.evaluator_relationship not in EVALUATOR_RELATIONSHIPS:
            raise ConversionError(
                f"evaluator_relationship must be one of {list(EVALUATOR_RELATIONSHIPS)}, "
                f"got {self.evaluator_relationship!r}"
            )
        if self.dataset_urls and self.hf_repo:
            raise ConversionError("give either dataset_urls or hf_repo, not both")
        if self.hf_split and not self.hf_repo:
            raise ConversionError("hf_split needs hf_repo")


# --------------------------------------------------------------------------- stats


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_sd(values: Sequence[float], mean: float) -> float:
    """Sample standard deviation (n - 1 denominator). Needs at least 2 values."""
    return math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _uncertainty(values: Sequence[float], repeated_cases: bool) -> Optional[Dict[str, Any]]:
    """Standard error of the mean, or None when it cannot be computed honestly.

    SE = sample SD / sqrt(n) treats the observations as independent items. That
    is false when the same test case contributes several observations (EvalPort
    ``attempt``), so in that case, and for fewer than two observations, no
    uncertainty is reported at all.
    """
    n = len(values)
    if n < 2 or repeated_cases:
        return None
    sd = _sample_sd(values, _mean(values))
    return {
        "standard_error": {"value": sd / math.sqrt(n), "method": STANDARD_ERROR_METHOD},
        "standard_deviation": sd,
        "num_samples": n,
    }


# ------------------------------------------------------------------------- helpers


def _epoch_seconds(timestamp: str, what: str) -> int:
    """RFC 3339 timestamp (offset required) -> integer Unix epoch seconds."""
    text = timestamp.strip()
    if text[-1:] in ("Z", "z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ConversionError(f"{what} {timestamp!r} is not an ISO 8601 date-time") from exc
    if parsed.tzinfo is None:
        # A naive timestamp would take the converter host's timezone and make
        # evaluation_id differ between machines.
        raise ConversionError(f"{what} {timestamp!r} has no UTC offset")
    return int(parsed.timestamp())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _json_str(value: Any) -> str:
    """EEE ``additional_details`` maps are str -> str, so non-strings are JSON-encoded."""
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _source_data(suite: Dict[str, Any], result_set: Dict[str, Any], ctx: EEEContext) -> Dict[str, Any]:
    dataset_name = suite.get("name") or suite["id"]
    details: Dict[str, str] = {"evalport_suite_id": suite["id"]}
    if isinstance(result_set.get("suite_version"), str):
        details["evalport_suite_version"] = result_set["suite_version"]
    if isinstance(suite.get("test_cases"), list):
        details["evalport_test_case_count"] = str(len(suite["test_cases"]))
    if ctx.dataset_urls:
        return {
            "dataset_name": dataset_name,
            "source_type": "url",
            "url": list(ctx.dataset_urls),
            "additional_details": details,
        }
    if ctx.hf_repo:
        data: Dict[str, Any] = {
            "dataset_name": dataset_name,
            "source_type": "hf_dataset",
            "hf_repo": ctx.hf_repo,
            "additional_details": details,
        }
        if ctx.hf_split:
            data["hf_split"] = ctx.hf_split
        return data
    # An EvalPort suite carries its test cases inline; it has no public dataset id.
    return {"dataset_name": dataset_name, "source_type": "other", "additional_details": details}


def _generation_config(result_set: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    provider = result_set.get("provider") or {}
    args: Dict[str, Any] = {}
    if _is_number(provider.get("temperature")):
        args["temperature"] = provider["temperature"]
    # EEE: max_tokens is an integer >= 1.
    if isinstance(provider.get("max_tokens"), int) and not isinstance(provider["max_tokens"], bool) and provider["max_tokens"] >= 1:
        args["max_tokens"] = provider["max_tokens"]
    extra = provider.get("extra")
    config: Dict[str, Any] = {}
    if args:
        config["generation_args"] = args
    if isinstance(extra, dict) and extra:
        config["additional_details"] = {f"provider_extra.{k}": _json_str(v) for k, v in sorted(extra.items())}
    return config or None


@dataclass
class _GraderTally:
    type: str
    scores: List[float] = field(default_factory=list)
    passed: int = 0
    unscored: int = 0
    cases_scored: Dict[str, int] = field(default_factory=dict)


def _tally(result_set: Dict[str, Any]) -> Dict[str, _GraderTally]:
    """Per-grader tallies. A null score is counted in ``unscored`` and nowhere else."""
    tallies: Dict[str, _GraderTally] = {}
    for result in result_set.get("results", []):
        case_id = result.get("test_case_id")
        for gr in result.get("grader_results", []):
            gid = gr.get("grader_id")
            if not isinstance(gid, str) or not gid:
                raise ConversionError(f"a grader_result in test case {case_id!r} has no grader_id")
            tally = tallies.setdefault(gid, _GraderTally(type=str(gr.get("type"))))
            if str(gr.get("type")) != tally.type:
                raise ConversionError(
                    f"grader {gid!r} appears with two types: {tally.type!r} and {gr.get('type')!r}"
                )
            score = gr.get("score")
            if score is None:
                tally.unscored += 1
                continue
            if not _is_number(score) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ConversionError(
                    f"grader {gid!r} in test case {case_id!r} has score {score!r}; "
                    "EvalPort scores are null or a number in [0, 1]"
                )
            tally.scores.append(float(score))
            tally.passed += 1 if gr.get("passed") is True else 0
            tally.cases_scored[case_id] = tally.cases_scored.get(case_id, 0) + 1
    return tallies


def _grader_results(
    suite: Dict[str, Any],
    tallies: Dict[str, _GraderTally],
    source_data: Dict[str, Any],
    generation_config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    descriptions = {g["id"]: g.get("description") for g in suite.get("graders", []) if isinstance(g, dict) and "id" in g}
    out: List[Dict[str, Any]] = []
    for gid in sorted(tallies):
        t = tallies[gid]
        if not t.scores:
            continue  # zero non-null scores: no numeric score is emitted (see module docstring)
        n = len(t.scores)
        details = {
            "scored_count": str(n),
            "unscored_count": str(t.unscored),
            "passed_count": str(t.passed),
            "pass_rate_over_scored": str(t.passed / n),
        }
        score_details: Dict[str, Any] = {"score": _mean(t.scores), "details": details}
        repeated = any(c > 1 for c in t.cases_scored.values())
        uncertainty = _uncertainty(t.scores, repeated)
        if uncertainty is not None:
            score_details["uncertainty"] = uncertainty
        elif repeated:
            details["uncertainty_omitted"] = "test cases have repeated attempts; observations are not independent"
        else:
            details["uncertainty_omitted"] = "fewer than 2 scored results"
        description = f"Mean score of EvalPort grader '{gid}' (type {t.type}) over its non-null results"
        if descriptions.get(gid):
            description += f": {descriptions[gid]}"
        result: Dict[str, Any] = {
            "evaluation_result_id": f"{suite['id']}/{gid}/mean_score",
            "evaluation_name": suite["id"],
            "source_data": source_data,
            "metric_config": {
                "evaluation_description": description,
                "metric_id": f"evalport.grader.{gid}",
                "metric_name": f"{gid} mean score",
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0.0,
                "max_score": 1.0,
                "additional_details": {
                    "aggregation": "mean_over_non_null_scores",
                    "aggregation_level": "grader",
                    "evalport_grader_id": gid,
                    "evalport_grader_type": t.type,
                    # Not a registry id: it claims no cross-source identity.
                    "metric_id_status": "unregistered",
                },
            },
            "score_details": score_details,
        }
        if generation_config:
            result["generation_config"] = generation_config
        out.append(result)
    return out


def _case_pass_rate_result(
    suite: Dict[str, Any],
    result_set: Dict[str, Any],
    source_data: Dict[str, Any],
    generation_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Suite-level share of test cases that passed, over cases with a verdict.

    A test case whose graders are all null-scored has no verdict (Rule 6) and is
    left out of numerator and denominator; it is counted in ``unscored_cases``.
    """
    outcomes: List[float] = []
    seen: Dict[str, int] = {}
    unscored_cases = 0
    for result in result_set.get("results", []):
        graders = result.get("grader_results", [])
        # No grader result at all, or only null scores: no verdict for this case.
        if not graders or all(g.get("score") is None for g in graders):
            unscored_cases += 1
            continue
        outcomes.append(1.0 if result.get("passed") is True else 0.0)
        seen[result["test_case_id"]] = seen.get(result["test_case_id"], 0) + 1
    if not outcomes:
        return None
    n = len(outcomes)
    passed = int(sum(outcomes))
    details = {
        "scored_cases": str(n),
        "unscored_cases": str(unscored_cases),
        "passed_cases": str(passed),
    }
    score_details: Dict[str, Any] = {"score": passed / n, "details": details}
    repeated = any(c > 1 for c in seen.values())
    uncertainty = _uncertainty(outcomes, repeated)
    if uncertainty is not None:
        score_details["uncertainty"] = uncertainty
    else:
        details["uncertainty_omitted"] = (
            "test cases have repeated attempts; observations are not independent"
            if repeated
            else "fewer than 2 scored cases"
        )
    result_out: Dict[str, Any] = {
        "evaluation_result_id": f"{suite['id']}/test_case_pass_rate",
        "evaluation_name": suite["id"],
        "source_data": source_data,
        "metric_config": {
            "evaluation_description": (
                "Share of EvalPort test cases whose overall 'passed' is true, over test cases "
                "with at least one non-null grader score"
            ),
            "metric_id": "evalport.test_case_pass_rate",
            "metric_name": "test case pass rate",
            "metric_unit": "proportion",
            "lower_is_better": False,
            "score_type": "continuous",
            "min_score": 0.0,
            "max_score": 1.0,
            "additional_details": {
                "aggregation": "share_of_scored_test_cases_passed",
                "aggregation_level": "suite",
                "metric_id_status": "unregistered",
            },
        },
        "score_details": score_details,
    }
    if generation_config:
        result_out["generation_config"] = generation_config
    return result_out


# ---------------------------------------------------------------------- conversion


def to_eee(suite: Dict[str, Any], result_set: Dict[str, Any], context: EEEContext) -> Dict[str, Any]:
    """Return one EEE (schema 0.3.0) aggregate record for one suite and one run."""
    context.validate()
    if result_set.get("suite_id") != suite.get("id"):
        raise ConversionError(
            f"ResultSet.suite_id {result_set.get('suite_id')!r} does not match Suite.id {suite.get('id')!r}"
        )
    results = result_set.get("results")
    if not isinstance(results, list) or not results:
        raise ConversionError("ResultSet.results is empty; there is nothing to export")
    if not isinstance(result_set.get("run_id"), str) or not isinstance(result_set.get("started_at"), str):
        raise ConversionError("ResultSet needs run_id and started_at")
    if isinstance(suite.get("test_cases"), list):
        known = {tc.get("id") for tc in suite["test_cases"] if isinstance(tc, dict)}
        unknown = sorted({str(r.get("test_case_id")) for r in results} - known)
        if unknown:
            raise ConversionError(f"ResultSet refers to test cases that are not in the suite: {unknown}")

    started = _epoch_seconds(result_set["started_at"], "ResultSet.started_at")
    source_data = _source_data(suite, result_set, context)
    generation_config = _generation_config(result_set)
    tallies = _tally(result_set)

    evaluation_results: List[Dict[str, Any]] = []
    case_result = _case_pass_rate_result(suite, result_set, source_data, generation_config)
    if case_result is not None:
        evaluation_results.append(case_result)
    evaluation_results += _grader_results(suite, tallies, source_data, generation_config)
    if not evaluation_results:
        raise ConversionError(
            "no grader produced a non-null score, so there is no number to report; "
            "refusing to write an empty EEE record"
        )

    provider = result_set.get("provider") or {}
    runner = result_set.get("runner") or {}

    model_details: Dict[str, str] = {
        "deployment_type": context.deployment_type,
        "model_availability": context.model_availability,
    }
    if provider.get("model") and provider["model"] != context.model_id:
        model_details["evalport_provider_model"] = str(provider["model"])
    model_info: Dict[str, Any] = {
        "name": context.model_name or provider.get("model") or context.model_id,
        "id": context.model_id,
        "additional_details": model_details,
    }
    if context.developer:
        model_info["developer"] = context.developer
    if context.inference_platform:
        model_info["inference_platform"] = context.inference_platform

    source_details: Dict[str, str] = {
        "input_format": "evalport_resultset",
        "evalport_spec_version": str(result_set.get("version", "")),
        "evalport_run_id": result_set["run_id"],
        "evalport_suite_id": suite["id"],
        "evalport_result_count": str(len(results)),
        "evalport_unscored_grader_results": str(sum(t.unscored for t in tallies.values())),
        "evalport_errored_results": str(sum(1 for r in results if r.get("error"))),
    }
    no_number = sorted(g for g, t in tallies.items() if not t.scores)
    if no_number:
        source_details["graders_without_scored_results"] = json.dumps(no_number)
    source_metadata: Dict[str, Any] = {
        "source_name": runner.get("name") or "EvalPort",
        "source_type": "evaluation_run",
        "source_organization_name": context.source_organization_name,
        "evaluator_relationship": context.evaluator_relationship,
        "additional_details": source_details,
    }
    if context.source_organization_url:
        source_metadata["source_organization_url"] = context.source_organization_url

    retrieved = context.retrieved_timestamp if context.retrieved_timestamp is not None else f"{time.time():.6f}"
    return {
        "schema_version": EEE_SCHEMA_VERSION,
        # Keyed on the evaluation's own start time, not on "now", so converting
        # the same run twice yields the same evaluation_id.
        "evaluation_id": f"{suite['id']}/{context.model_id}/{started}",
        "evaluation_timestamp": str(started),
        "retrieved_timestamp": retrieved,
        "source_metadata": source_metadata,
        "eval_library": {
            "name": context.eval_library_name or runner.get("name") or "unknown",
            "version": context.eval_library_version or runner.get("version") or "unknown",
        },
        "model_info": model_info,
        "evaluation_results": evaluation_results,
    }


# --------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert an EvalPort Suite + ResultSet into one Every Eval Ever (schema 0.3.0) record.",
        epilog="EvalPort does not carry the model identity, deployment facts or who ran the evaluation, "
        "so the required options below must be given; none is guessed.",
    )
    p.add_argument("suite", help="EvalPort Suite JSON file")
    p.add_argument("resultset", help="EvalPort ResultSet JSON file")
    req = p.add_argument_group("required EEE context (not in EvalPort)")
    req.add_argument("--model-id", required=True, help="EEE model_info.id, e.g. openai/gpt-4o-2024-11-20")
    req.add_argument("--deployment-type", required=True, choices=DEPLOYMENT_TYPES)
    req.add_argument("--model-availability", required=True, choices=MODEL_AVAILABILITIES)
    req.add_argument("--source-organization-name", required=True, help="organization providing the data")
    req.add_argument("--evaluator-relationship", required=True, choices=EVALUATOR_RELATIONSHIPS,
                     help="relationship of the evaluator to the model developer")
    opt = p.add_argument_group("optional EEE context")
    opt.add_argument("--model-name", help="default: ResultSet.provider.model, else --model-id")
    opt.add_argument("--developer", help="model_info.developer (never derived from --model-id)")
    opt.add_argument("--inference-platform")
    opt.add_argument("--source-organization-url")
    opt.add_argument("--eval-library-name", help="default: ResultSet.runner.name, else 'unknown'")
    opt.add_argument("--eval-library-version", help="default: ResultSet.runner.version, else 'unknown'")
    opt.add_argument("--dataset-url", action="append", default=[], help="where the evaluated data lives (repeatable)")
    opt.add_argument("--hf-repo", help="Hugging Face dataset repo of the evaluated data")
    opt.add_argument("--hf-split")
    opt.add_argument("--retrieved-timestamp", help="Unix epoch string for record creation (default: now)")
    p.add_argument("-o", "--output", help="write the record here instead of stdout")
    return p


def _load(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConversionError(f"{path} is not a JSON object")
    return data


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        ctx = EEEContext(
            model_id=args.model_id,
            deployment_type=args.deployment_type,
            model_availability=args.model_availability,
            source_organization_name=args.source_organization_name,
            evaluator_relationship=args.evaluator_relationship,
            model_name=args.model_name,
            developer=args.developer,
            inference_platform=args.inference_platform,
            source_organization_url=args.source_organization_url,
            eval_library_name=args.eval_library_name,
            eval_library_version=args.eval_library_version,
            dataset_urls=args.dataset_url,
            hf_repo=args.hf_repo,
            hf_split=args.hf_split,
            retrieved_timestamp=args.retrieved_timestamp,
        )
        record = to_eee(_load(args.suite), _load(args.resultset), ctx)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    # allow_nan=False: EEE's strict reader rejects NaN / Infinity tokens.
    text = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
