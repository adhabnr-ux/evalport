"""Convert MMMU (https://github.com/MMMU-Benchmark/MMMU) dataset samples and
graded evaluation runs to/from the EvalPort open evaluation format.

MMMU ("A Massive Multi-discipline Multimodal Understanding and Reasoning
Benchmark for Expert AGI") ships its own evaluation surface in
``mmmu/main_eval_only.py`` and ``mmmu/utils/{data_utils,eval_utils}.py``, read
directly (not guessed) before writing this module:

- A dataset sample, after ``data_utils.process_single_sample()``, is a dict
  shaped ``{"id", "question", "options", "answer", "image", "question_type"}``
  — ``question_type`` is either ``"multiple-choice"`` (options is a
  stringified Python list of choice texts, answer is a letter like ``"A"``)
  or ``"open"`` (answer is a ground-truth string or list of acceptable
  strings/numbers). ``to_openeval()``/``from_openeval()`` below convert this
  dataset-level shape to/from an EvalPort ``Suite``.

- ``main_eval_only.py`` builds ``exampels_to_eval`` entries shaped
  ``{"id", "question_type", "answer", "parsed_pred"}`` per graded example
  (``parsed_pred`` already run through ``eval_utils.parse_open_response()``
  for open questions), calls ``eval_utils.evaluate(exampels_to_eval)`` to get
  a ``judge_dict`` (``{id: "Correct"|"Wrong"}``), then rolls per-category
  ``metric_dict``/``num_example`` results up through
  ``DOMAIN_CAT2SUB_CAT``/``calculate_ins_level_acc`` into a
  ``printable_results`` dict keyed by category name and ``"Overall-<Domain>"``
  / ``"Overall"``, each ``{"num": int, "acc": float}``. ``result_to_openeval()``
  below converts that graded-run shape to an EvalPort ``ResultSet``, carrying
  ``judge_dict``'s per-item correctness into ``ResultSet.results`` and
  ``printable_results`` straight into ``ResultSet.summary`` — the exact
  per-discipline breakdown that is the point of MMMU (broad expert-level
  knowledge spanning six disciplines: Art & Design, Business, Science, Health
  & Medicine, Humanities & Social Science, Tech & Engineering), not just one
  aggregate accuracy number.

Grading: MMMU's own ``eval_multi_choice()`` is a plain equality check on the
predicted letter, which maps directly onto EvalPort's standard
``exact_match`` grader. ``eval_open()`` normalizes numbers (2-decimal
rounding, comma stripping) and does substring containment on lowercased
strings, which has no honest standard-grader equivalent, so it maps to
``custom`` with ``params.handler="mmmu:eval_open"`` — the same convention
every other adapter in this ecosystem uses for a grading rule it can't
natively represent, per the reference shape in
``adapters/autogen-openeval-adapter`` and issue #6.

No image binaries are embedded: MMMU dataset samples carry PIL Image objects
(or None) in ``image``/``image_1``, which are not JSON-serializable and not
portable across tools anyway. When an image is present, only a boolean
``metadata.mmmu.has_image`` flag is recorded (plus the field name it came
from) — callers that need the actual image keep it alongside the original
MMMU sample and join back on ``TestCase.id``.
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "result_to_openeval", "__version__"]
__version__ = "0.1.0"

_RESERVED_METADATA_KEY = "mmmu"

# Read verbatim from mmmu/utils/data_utils.py (DOMAIN_CAT2SUB_CAT) so
# category -> domain lookups in this module match MMMU's own grouping
# exactly, rather than re-deriving it and risking drift.
DOMAIN_CAT2SUB_CAT: Dict[str, List[str]] = {
    "Art and Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health and Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities and Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech and Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

_CAT2DOMAIN: Dict[str, str] = {
    cat: domain for domain, cats in DOMAIN_CAT2SUB_CAT.items() for cat in cats
}

_MULTI_CHOICE_GRADER: Dict[str, Any] = {"id": "gr_mmmu_multi_choice", "type": "exact_match"}
_OPEN_GRADER: Dict[str, Any] = {
    "id": "gr_mmmu_open",
    "type": "custom",
    "params": {
        "handler": "mmmu:eval_open",
        "description": (
            "MMMU's utils.eval_utils.eval_open(): normalizes numbers (comma "
            "strip, round to 2 decimals) and strings (lowercased), then "
            "checks containment of any normalized gold answer in the "
            "normalized parsed prediction."
        ),
    },
}


def _category_from_id(data_id: str) -> Optional[str]:
    """Recover the category name from an MMMU id like ``"val_Accounting_12"``
    or ``"validation_Clinical_Medicine_3"``, mirroring
    ``main_eval_only.py``'s own ``"_".join(data_id.split("_")[1:-1])``."""
    parts = data_id.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[1:-1])


def _coerce_options(options: Any) -> List[str]:
    """``options`` on a raw MMMU sample is a stringified Python list (HF
    datasets serialize it that way; ``data_utils.construct_prompt()`` reads
    it back with ``eval(sample['options'])``). Accept a real list too, so
    this module works whether or not the caller already parsed it."""
    if isinstance(options, list):
        return [str(o) for o in options]
    if isinstance(options, str) and options.strip():
        try:
            parsed = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return [str(o) for o in parsed]
    return []


def _index2ans(options: List[str]) -> Dict[str, str]:
    """Mirrors ``data_utils.get_multi_choice_info()``: 'A' -> first option, etc."""
    return {chr(ord("A") + i): opt for i, opt in enumerate(options)}


def to_openeval(
    samples: Sequence[Dict[str, Any]],
    *,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a list of raw MMMU dataset samples (post
    ``data_utils.process_single_sample()`` shape: ``id``, ``question``,
    ``options``, ``answer``, ``question_type``, and optionally ``image``/
    ``image_1``) into an EvalPort ``Suite`` (plain dict).

    One ``TestCase`` per sample. Multiple-choice samples get the options
    rendered into ``input`` (so a runner without MMMU's own prompt-building
    code can still ask the question) and graded with the standard
    ``exact_match`` grader; open questions are graded with a ``custom``
    grader identifying MMMU's own ``eval_open()`` semantics (see module
    docstring). Category and domain (via ``DOMAIN_CAT2SUB_CAT``) are derived
    from the sample id and recorded in metadata, matching how
    ``main_eval_only.py`` groups results for its own per-category/per-domain
    rollup.

    Pass the result to ``openeval.validate.validate_suite()`` to confirm
    compliance, or ``json.dump()`` it to share as a ``.json`` suite file.
    """
    if not samples:
        raise ValueError("samples is empty -- nothing to convert")

    test_cases: List[Dict[str, Any]] = []
    graders_seen: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        data_id = sample.get("id")
        if not data_id:
            raise ValueError(f"sample missing 'id': {sample!r}")
        data_id = str(data_id)
        question_type = sample.get("question_type", "open")
        question = sample.get("question", "")
        category = _category_from_id(data_id)
        domain = _CAT2DOMAIN.get(category) if category else None

        metadata: Dict[str, Any] = {
            _RESERVED_METADATA_KEY: {
                "question_type": question_type,
                "category": category,
                "domain": domain,
            }
        }

        image_field = "image_1" if sample.get("image_1") is not None else (
            "image" if sample.get("image") is not None else None
        )
        if image_field is not None:
            metadata[_RESERVED_METADATA_KEY]["has_image"] = True
            metadata[_RESERVED_METADATA_KEY]["image_field"] = image_field
            # Only carry the value through when it is already a plain string
            # (a path or URL) -- a PIL.Image.Image instance is not
            # JSON-serializable and is dropped rather than stringified into
            # something misleading (e.g. its repr/memory address).
            image_value = sample.get(image_field)
            if isinstance(image_value, str):
                metadata[_RESERVED_METADATA_KEY]["image"] = image_value

        test_case: Dict[str, Any] = {
            "id": data_id,
            "graders": [],
            "metadata": metadata,
        }

        if question_type == "multiple-choice":
            options = _coerce_options(sample.get("options"))
            index2ans = _index2ans(options)
            rendered = question
            for letter, option_text in index2ans.items():
                rendered += f"\n({letter}) {option_text}"
            test_case["input"] = rendered
            answer = sample.get("answer")
            if answer is not None:
                test_case["expected_output"] = str(answer)
            metadata[_RESERVED_METADATA_KEY]["options"] = options
            metadata[_RESERVED_METADATA_KEY]["index2ans"] = index2ans
            test_case["graders"] = [_MULTI_CHOICE_GRADER["id"]]
            graders_seen[_MULTI_CHOICE_GRADER["id"]] = _MULTI_CHOICE_GRADER
        else:
            test_case["input"] = question
            answer = sample.get("answer")
            if isinstance(answer, list):
                # EvalPort's expected_output is a single string per the spec
                # -- when MMMU's ground truth is itself a list of acceptable
                # answers, keep the full list in metadata for a custom
                # grader/human reviewer, and use the first as expected_output
                # so the field is never silently dropped.
                metadata[_RESERVED_METADATA_KEY]["acceptable_answers"] = [str(a) for a in answer]
                if answer:
                    test_case["expected_output"] = str(answer[0])
            elif answer is not None:
                test_case["expected_output"] = str(answer)
            test_case["graders"] = [_OPEN_GRADER["id"]]
            graders_seen[_OPEN_GRADER["id"]] = _OPEN_GRADER

        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": suite_id or "mmmu_suite",
        "test_cases": test_cases,
        "graders": list(graders_seen.values()),
        "metadata": {_RESERVED_METADATA_KEY: {"source": "mmmu"}},
    }
    if name is not None:
        suite["name"] = name
    if description is not None:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reverse ``to_openeval()``: recover MMMU-shaped sample dicts
    (``id``, ``question``, ``options``, ``answer``, ``question_type``) from
    an EvalPort ``Suite`` that this module produced.

    Test cases whose metadata doesn't carry this module's ``"mmmu"``
    namespace are skipped, matching the convention used across this repo's
    adapters for data they didn't originate.
    """
    samples: List[Dict[str, Any]] = []
    for test_case in suite.get("test_cases", []):
        metadata = (test_case.get("metadata") or {}).get(_RESERVED_METADATA_KEY)
        if not metadata:
            continue
        question_type = metadata.get("question_type", "open")
        raw_input = test_case.get("input", "")
        if question_type == "multiple-choice":
            # Strip the "\n(A) ..." lines to_openeval() appended, recovering
            # the bare question text.
            question = raw_input.split("\n(A) ")[0] if "\n(A) " in raw_input else raw_input
            options = metadata.get("options", [])
        else:
            question = raw_input
            options = []
        answer: Union[str, List[str], None]
        if "acceptable_answers" in metadata:
            answer = metadata["acceptable_answers"]
        else:
            answer = test_case.get("expected_output")
        samples.append(
            {
                "id": test_case.get("id"),
                "question": question,
                "options": options,
                "answer": answer,
                "question_type": question_type,
            }
        )
    return samples


def result_to_openeval(
    exampels_to_eval: Sequence[Dict[str, Any]],
    judge_dict: Dict[str, str],
    printable_results: Dict[str, Dict[str, Any]],
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one graded MMMU evaluation run into an EvalPort ``ResultSet``.

    Mirrors the three real objects ``mmmu/main_eval_only.py`` produces:

    - ``exampels_to_eval``: the list built right before ``evaluate()`` is
      called, each ``{"id", "question_type", "answer", "parsed_pred"}``.
    - ``judge_dict``: ``evaluate(exampels_to_eval)``'s first return value,
      ``{id: "Correct" | "Wrong"}``.
    - ``printable_results``: the per-category/per-domain rollup built at the
      bottom of ``main_eval_only.py`` from ``DOMAIN_CAT2SUB_CAT`` and
      ``calculate_ins_level_acc()`` -- keys are category names (e.g.
      ``"Accounting"``), ``"Overall-<Domain>"`` (e.g.
      ``"Overall-Business"``), and ``"Overall"``, each
      ``{"num": int, "acc": float}``.

    One ``Result`` per entry in ``exampels_to_eval``, correctness taken from
    ``judge_dict`` (never re-derived — this module does not re-implement
    MMMU's ``eval_multi_choice``/``eval_open`` comparison logic, it only
    carries the real verdict through). ``printable_results`` is preserved
    verbatim as ``ResultSet.summary``, so the per-discipline breakdown MMMU
    is designed around survives the round trip untouched.
    """
    if not exampels_to_eval:
        raise ValueError("exampels_to_eval is empty -- nothing to convert")

    results: List[Dict[str, Any]] = []
    for example in exampels_to_eval:
        data_id = str(example["id"])
        question_type = example.get("question_type", "open")
        verdict = judge_dict.get(data_id)
        if verdict is None:
            raise ValueError(f"judge_dict has no verdict for id={data_id!r}")
        passed = verdict == "Correct"

        parsed_pred = example.get("parsed_pred")
        if isinstance(parsed_pred, list):
            actual_output = ", ".join(str(p) for p in parsed_pred)
        elif parsed_pred is not None:
            actual_output = str(parsed_pred)
        else:
            actual_output = None

        if question_type == "multiple-choice":
            grader_id, grader_type = _MULTI_CHOICE_GRADER["id"], "exact_match"
        else:
            grader_id, grader_type = _OPEN_GRADER["id"], "custom"

        result: Dict[str, Any] = {
            "test_case_id": data_id,
            "passed": passed,
            "grader_results": [
                {
                    "grader_id": grader_id,
                    "type": grader_type,
                    "score": 1.0 if passed else 0.0,
                    "passed": passed,
                }
            ],
        }
        if actual_output is not None:
            result["actual_output"] = actual_output
        results.append(result)

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": dict(printable_results),
        "metadata": {_RESERVED_METADATA_KEY: {"source": "mmmu"}},
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    return result_set
