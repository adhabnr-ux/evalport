"""Convert between Arize Phoenix datasets/experiments and EvalPort.

EvalPort (https://github.com/adhabnr-ux/evalport) is an open interchange
format (Apache 2.0) for portable LLM evaluation datasets: test cases,
graders, suites, and results as plain JSON, shared across evaluation tools.

Arize Phoenix (https://github.com/Arize-ai/phoenix, `pip install
arize-phoenix-client`) represents dataset rows as ``v1.DatasetExample``
TypedDicts -- ``{"id", "input", "output", "metadata", ...}`` where ``input``
and ``output`` are themselves arbitrary JSON mappings (e.g.
``{"question": "..."}`` / ``{"answer": "..."}``), not flat strings. Completed
experiment runs come back as a ``RanExperiment`` TypedDict: a list of
``ExperimentRun`` task outputs plus a list of ``ExperimentEvaluationRun``
evaluator results, joined by run id.

Three entry points, matching the shape used by every other adapter in the
EvalPort ecosystem:

    to_openeval(examples, ...)
        ``Dataset.examples`` / any iterable of ``v1.DatasetExample``-shaped
        objects -> an EvalPort suite.

    from_openeval(suite, ...)
        An EvalPort suite's test cases -> a list of dicts shaped for
        ``client.datasets.create_dataset(examples=...)``.

    experiment_to_openeval(ran_experiment, ...)
        A completed ``RanExperiment`` (task_runs + evaluation_runs) -> an
        EvalPort ResultSet, one GraderResult per evaluator that scored each
        example.

Because Phoenix's ``input``/``output`` are arbitrary mappings rather than
plain strings, and EvalPort's ``TestCase.input``/``expected_output`` must be
strings (or an array of strings for multi-turn), key detection here works
the same way ``opik-openeval-adapter`` handles Opik's schema-less dataset
items: check common key names, fall back to the mapping's only key if it has
exactly one, and fall back to a JSON dump of the whole mapping as a last
resort so nothing is ever silently dropped. The full raw ``input``/``output``
mappings are always preserved under ``metadata["phoenix"]`` regardless of
which key was picked, so a lossy heuristic guess never loses data.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from openeval.version import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "experiment_to_openeval"]

DEFAULT_INPUT_KEYS = ("input", "question", "query", "prompt", "user_input")
DEFAULT_OUTPUT_KEYS = (
    "expected_output",
    "output",
    "answer",
    "reference",
    "ground_truth",
)


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a plain dict (v1.DatasetExample /
    v1.ExperimentRun, both TypedDicts) or an attribute-style object
    (ExampleProxy, or a user's own wrapper) -- Phoenix's own client code
    supports both access styles for the same underlying data, so this
    adapter does too rather than forcing one."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _extract_text(
    mapping: Optional[Mapping[str, Any]], preferred_keys: Tuple[str, ...]
) -> Optional[str]:
    """Pick a single string out of a Phoenix input/output mapping.

    Tries the preferred key names first (in order), then falls back to the
    mapping's only key if it has exactly one, then falls back to a JSON dump
    of the whole mapping so nothing unrecognized is silently dropped.
    """
    if not mapping:
        return None
    for key in preferred_keys:
        if key in mapping:
            return _stringify(mapping[key])
    if len(mapping) == 1:
        return _stringify(next(iter(mapping.values())))
    return _stringify(dict(mapping))


def to_openeval(
    examples: Iterable[Any],
    suite_id: str = "phoenix_dataset",
    input_key: Optional[str] = None,
    expected_output_key: Optional[str] = None,
    grader_type: str = "llm_judge",
) -> Dict[str, Any]:
    """Convert Phoenix dataset examples into an EvalPort suite.

    Args:
        examples: An iterable of ``v1.DatasetExample``-shaped objects --
            works with ``Dataset.examples`` directly (Phoenix's ``Dataset``
            is itself iterable-friendly via this attribute), a raw list of
            dicts with ``id``/``input``/``output``/``metadata`` keys, or
            ``ExampleProxy`` instances (attribute access). Each example's
            ``input``/``output`` are themselves mappings, e.g.
            ``{"question": "..."}`` -- not flat strings.
        suite_id: The EvalPort ``EvalSuite.id``.
        input_key / expected_output_key: Force which key inside each
            example's ``input``/``output`` mapping to use, overriding the
            auto-detection heuristic (checks
            ``question``/``query``/``prompt``/``user_input`` for input and
            ``output``/``answer``/``reference``/``ground_truth`` for
            expected output, then falls back to the mapping's only key, then
            to a full JSON dump).
        grader_type: ``"exact_match"`` or ``"llm_judge"`` (default) for the
            single default grader attached to every test case, mirroring
            every other adapter in this ecosystem -- EvalPort requires
            ``graders`` to be non-empty per test case, and Phoenix examples
            don't carry a grader definition of their own.

    Returns:
        A dict matching EvalPort's EvalSuite schema
        (validate with ``openeval.validate.validate_suite``).
    """
    examples = list(examples)
    if not examples:
        return {
            "version": OPENEVAL_VERSION,
            "id": suite_id,
            "graders": [],
            "test_cases": [],
        }

    if grader_type == "exact_match":
        graders = [
            {"id": "gr_output_match", "type": "exact_match", "params": {"ignore_case": True}}
        ]
    else:
        graders = [
            {
                "id": "gr_output_match",
                "type": "llm_judge",
                "params": {
                    "prompt": "Does the actual output '{output}' match the expected output '{expected_output}'? Answer yes or no.",
                    "model": "gpt-4o",
                },
            }
        ]
    grader_ids = [g["id"] for g in graders]

    input_keys = (input_key,) if input_key else DEFAULT_INPUT_KEYS
    output_keys = (expected_output_key,) if expected_output_key else DEFAULT_OUTPUT_KEYS

    test_cases = []
    for example in examples:
        example_id = _field(example, "id")
        raw_input = _field(example, "input") or {}
        raw_output = _field(example, "output") or {}
        raw_metadata = _field(example, "metadata") or {}

        input_text = _extract_text(raw_input, input_keys) or ""
        expected_output = _extract_text(raw_output, output_keys)

        test_case: Dict[str, Any] = {
            "id": str(example_id) if example_id is not None else input_text[:64],
            "input": input_text,
            "graders": list(grader_ids),
            "metadata": {
                "phoenix": {
                    "example_id": example_id,
                    "raw_input": dict(raw_input),
                    "raw_output": dict(raw_output),
                    **({"raw_metadata": dict(raw_metadata)} if raw_metadata else {}),
                }
            },
        }
        if expected_output is not None:
            test_case["expected_output"] = expected_output

        test_cases.append(test_case)

    return {
        "version": OPENEVAL_VERSION,
        "id": suite_id,
        "graders": graders,
        "test_cases": test_cases,
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite's test cases into Phoenix dataset examples.

    Returns a list of dicts shaped for
    ``client.datasets.create_dataset(name=..., examples=...)`` (Phoenix's own
    upload shape: ``{"input": {...}, "output": {...}, "metadata": {...},
    "id": ...}``). ``input``/``output`` are wrapped as single-key mappings
    (``{"input": ...}`` / ``{"expected_output": ...}``) using the same key
    names ``to_openeval``'s auto-detection checks first, so a
    suite -> Phoenix -> ``to_openeval`` round trip recovers the exact text
    without needing ``input_key``/``expected_output_key`` overrides.
    """
    test_cases = suite.get("test_cases") or []
    examples = []
    for tc in test_cases:
        entry: Dict[str, Any] = {
            "id": tc.get("id"),
            "input": {"input": tc.get("input")},
            "output": {"expected_output": tc.get("expected_output")}
            if tc.get("expected_output") is not None
            else {},
        }
        metadata = tc.get("metadata") or {}
        if metadata:
            entry["metadata"] = metadata
        examples.append(entry)
    return examples


def _normalize_evaluation_results(result: Any) -> List[Dict[str, Any]]:
    """ExperimentEvaluationRun.result is either a single
    ExperimentEvaluation dict or a Sequence of them (is_evaluation_result /
    is_score_result in phoenix.client.resources.experiments.types)."""
    if result is None:
        return []
    if isinstance(result, Mapping):
        return [dict(result)]
    if isinstance(result, (list, tuple)):
        return [dict(r) if isinstance(r, Mapping) else {"score": r} for r in result]
    return [{"score": result}]


def experiment_to_openeval(
    ran_experiment: Dict[str, Any],
    suite_id: Optional[str] = None,
    run_id: Optional[str] = None,
    pass_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Convert a completed Phoenix ``RanExperiment`` into an EvalPort
    ResultSet.

    Args:
        ran_experiment: A ``RanExperiment`` TypedDict -- as returned by
            ``phoenix.client.experiments.run_experiment()`` or
            ``client.experiments.get_experiment(experiment_id=...)``. Its
            ``task_runs`` are plain dicts (``v1.ExperimentRun``); its
            ``evaluation_runs`` are ``ExperimentEvaluationRun`` dataclass
            instances -- this function reads both correctly via ``_field``.
        suite_id: EvalPort ``ResultSet.suite_id``. Defaults to
            ``ran_experiment["dataset_id"]`` (the EvalPort suite these
            results correspond to, if it was built by ``to_openeval`` from
            the same dataset).
        run_id: EvalPort ``ResultSet.run_id``. Defaults to
            ``ran_experiment["experiment_id"]``.
        pass_threshold: A grader result with a numeric ``score`` passes when
            ``score >= pass_threshold``. A grader result with only a
            ``label`` (no score) passes when the label case-insensitively
            reads as an affirmative value (``"true"``, ``"pass"``,
            ``"passed"``, ``"correct"``, ``"yes"``, ``"good"``) -- Phoenix
            evaluators commonly return a label instead of (or alongside) a
            score. A result's overall ``passed`` follows the same
            convention every other EvalPort adapter uses: every one of its
            grader results must individually pass.

    Returns:
        A dict matching EvalPort's ResultSet schema (validate with
        ``openeval.validate.validate_result_set``).
    """
    task_runs = ran_experiment.get("task_runs") or []
    evaluation_runs = ran_experiment.get("evaluation_runs") or []

    evals_by_run_id: Dict[str, List[Any]] = {}
    for eval_run in evaluation_runs:
        key = _field(eval_run, "experiment_run_id")
        evals_by_run_id.setdefault(key, []).append(eval_run)

    results = []
    for task_run in task_runs:
        task_run_id = _field(task_run, "id")
        example_id = _field(task_run, "dataset_example_id")
        output = _field(task_run, "output")
        error = _field(task_run, "error")

        grader_results = []
        for eval_run in evals_by_run_id.get(task_run_id, []):
            eval_error = _field(eval_run, "error")
            eval_name = _field(eval_run, "name") or "phoenix_evaluator"
            for scored in _normalize_evaluation_results(_field(eval_run, "result")):
                score = scored.get("score")
                label = scored.get("label")
                grader_id = scored.get("name") or eval_name

                if score is not None:
                    score = max(0.0, min(1.0, float(score)))
                    passed = score >= pass_threshold
                elif label is not None:
                    passed = str(label).strip().lower() in (
                        "true",
                        "pass",
                        "passed",
                        "correct",
                        "yes",
                        "good",
                    )
                else:
                    passed = False

                grader_result: Dict[str, Any] = {
                    "grader_id": str(grader_id),
                    "type": "custom",
                    "score": score,
                    "passed": passed,
                }
                if scored.get("explanation"):
                    grader_result["reason"] = str(scored["explanation"])
                if eval_error:
                    grader_result["passed"] = False
                grader_results.append(grader_result)

        result: Dict[str, Any] = {
            "test_case_id": str(example_id),
            "grader_results": grader_results,
            "passed": (
                all(g["passed"] for g in grader_results) if grader_results else False
            ),
            "metadata": {"phoenix": {"experiment_run_id": task_run_id}},
        }
        if output is not None:
            result["actual_output"] = _stringify(output)
        if error:
            result["error"] = {"type": "runner_error", "message": str(error)}

        start_time = _field(task_run, "start_time")
        end_time = _field(task_run, "end_time")
        if start_time and end_time:
            from datetime import datetime

            try:
                start_dt = datetime.fromisoformat(str(start_time).replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(str(end_time).replace("Z", "+00:00"))
                result["duration_ms"] = max(
                    0, round((end_dt - start_dt).total_seconds() * 1000)
                )
            except ValueError:
                pass

        results.append(result)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id or str(ran_experiment.get("dataset_id") or "phoenix_dataset"),
        "run_id": run_id or str(ran_experiment.get("experiment_id") or "phoenix_experiment"),
        "started_at": now_iso,
        "completed_at": now_iso,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": (passed / total) if total else 0.0,
        },
    }
