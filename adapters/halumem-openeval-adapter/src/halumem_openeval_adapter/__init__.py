"""Convert MemTensor/HaluMem (https://github.com/MemTensor/HaluMem) operation-level
evaluation records to and from EvalPort (https://github.com/adhabnr-ux/evalport), the
open interchange format for portable LLM evaluation test cases, graders, suites, and
results.

Built following the design worked out in MemTensor/HaluMem#12
(https://github.com/MemTensor/HaluMem/issues/12), addressing the four points the
HaluMem maintainer (@hush-cd) raised on the initial proposal:

1. HaluMem's verdicts (`result_type`, `memory_update_type`) come from HaluMem's own
   LLM-based evaluator (`eval/eval_tools.py`, via `eval/llms.py`'s
   `llm_request_for_json`), not a human grader -- every ``GraderResult`` this module
   emits uses ``type: "llm_judge"``, never ``"human"``. The judge model is read from
   the real ``OPENAI_MODEL`` environment variable HaluMem's own `eval/llms.py` uses
   (``MODEL = os.getenv('OPENAI_MODEL')``) unless a ``judge_model`` argument overrides
   it -- there is no hardcoded default model name anywhere in this module.
2. Categorical outcomes are never collapsed into a bare ``score``. Each grader result's
   ``reason`` is the literal HaluMem verdict string, and the same value is duplicated
   onto a stable ``metadata.halumem.<field>`` key (``result_type`` for QA,
   ``memory_update_type`` for updates, which additionally has the ``"Other"`` outcome
   HaluMem's update task uses that the QA task does not) so a consumer never has to
   parse `reason` text to recover Hallucination vs. Omission vs. Other. ``score``/
   ``passed`` are a derived convenience view on top of that canonical field, not the
   source of truth.
3. Extraction scoring semantics are preserved verbatim: ``memory_integrity_score``,
   ``memory_accuracy_score``, ``is_included_in_golden_memories``, ``importance``,
   ``memory_source`` (which is how HaluMem marks an "interference"/distractor memory),
   and every other real field on a HaluMem record land untouched in
   ``metadata.halumem.*``. ``tests/test_adapter.py`` includes a round-trip test that
   recomputes HaluMem's own ``memory_integrity`` / ``memory_accuracy`` /
   ``memory_extraction_f1`` / ``memory_update`` aggregate metrics purely from converted
   EvalPort ``ResultSet`` objects and diffs them against a real
   ``eval_results["overall_score"]`` shape computed the same way
   ``eval/evaluation.py``'s ``aggregate_eval_results()`` computes it.
4. Test-case IDs are a stable SHA-256 digest over HaluMem's own identifiers (``uuid``,
   ``ssession_id``, and the record's real content field), never Python's built-in
   ``hash()`` -- see ``_stable_id()``. Same input always produces the same ID, across
   runs and across interpreter invocations (``hash()`` on a fresh Python process is
   randomized per-process for strings unless ``PYTHONHASHSEED`` is fixed; a real digest
   has no such caveat).

## What this module operates on

HaluMem's real evaluation output (``eval/evaluation.py``'s ``main()``) is a single
JSON object -- not JSONL -- shaped like::

    {
      "overall_score": {...},
      "memory_integrity_records": [...],
      "memory_accuracy_records": [...],
      "memory_update_records": [...],
      "question_answering_records": [...]
    }

Every function below accepts either that whole dict (and pulls the right list out by
key) or an already-extracted ``list[dict]`` of just that operation's records --
whichever is more convenient for the caller. ``load_eval_results()`` is a thin
convenience wrapper for reading that JSON file from disk.

## The four HaluMem operations, and what round-trips

Confirmed against the real ``eval/evaluation.py`` / ``eval/eval_tools.py`` on the
HaluMem ``main`` branch (not guessed from the README):

- **``"qa"``** (``question_answering_records``): each record already carries
  ``question`` / ``answer`` / ``evidence`` (the pre-run test case shape) *and*, once
  scored, ``system_response`` / ``result_type`` (``"Correct"`` | ``"Hallucination"`` |
  ``"Omission"``). ``to_openeval(..., "qa")`` uses the pre-run fields;
  ``result_to_openeval(..., "qa")`` uses the post-run fields. This is the most
  complete round trip of the four operations.
- **``"memory_integrity"``** (``memory_integrity_records``): each record is one golden
  "expected memory point" plus its ``memory_integrity_score`` (0/1/2, where 2 = fully
  captured) after grading against the system's full extracted-memory pool for that
  session. **Known real-data limitation, stated plainly rather than papered over:**
  the extracted-memory pool that was graded against (``extract_memories_str`` in
  ``evaluation.py``) is *not itself* persisted onto the record -- only the golden
  point and its score are. This module therefore cannot reconstruct an
  ``actual_output`` for this operation from ``eval_results`` alone; ``TestCase.input``
  is the real golden ``memory_content`` field, and ``Result.actual_output`` is left
  unset rather than fabricated. A record with ``memory_source == "interference"`` is a
  distractor the system should *not* have recalled, so ``passed`` for those records
  means ``memory_integrity_score == 0`` (not 2) -- the same asymmetry
  ``aggregate_eval_results()`` itself encodes (see its separate
  ``interference_memory_scores`` counter).
- **``"memory_accuracy"``** (``memory_accuracy_records``): each record is one
  system-*extracted* candidate memory plus its ``memory_accuracy_score`` (0/1/2, where
  2 = fully supported/no hallucination) and ``is_included_in_golden_memories``
  (``"true"``/``"false"``, kept as the literal string HaluMem uses, since it is not
  reliably boolean-typed in real output). Same limitation as above: the dialogue and
  golden-memories text used for grading are not persisted on the record itself.
- **``"memory_update"``** (``memory_update_records``): each record is one golden
  "target updated memory" plus ``original_memories`` (the pre-update state, a real
  list field -- used as ``TestCase.context``), ``memories_from_system`` (the system's
  post-update output, when present -- used as ``Result.actual_output``, unlike the two
  extraction operations above, since HaluMem *does* persist this one), and
  ``memory_update_type`` (``"Correct"`` | ``"Hallucination"`` | ``"Omission"`` |
  ``"Other"``).

## Grader prompts

Every well-known ``llm_judge`` grader type in the EvalPort spec (``spec/SPEC.md``,
Validation Rule 4) requires ``params.model`` and a ``params.prompt`` containing at
least one of ``{input}``/``{expected}``/``{output}``. HaluMem's real prompts (the
``EVALUATION_PROMPT_FOR_*`` constants in ``eval/eval_tools.py``) use different
placeholder names (``{memories}``, ``{expected_memory_point}``, ``{question}``, etc.)
that don't match that convention, so this module's grader ``params.prompt`` values are
condensed paraphrases of HaluMem's real rubric text, rewritten only enough to use
EvalPort's placeholder convention -- not independently invented criteria. Each is
labeled in a comment with which real prompt constant it's adapted from, so it can be
checked against the source.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

__all__ = [
    "to_openeval",
    "from_openeval",
    "result_to_openeval",
    "load_eval_results",
    "OPERATIONS",
]

_SPEC_VERSION = "1.0.0"

OPERATIONS = ("qa", "memory_integrity", "memory_accuracy", "memory_update")

_RECORD_KEY = {
    "qa": "question_answering_records",
    "memory_integrity": "memory_integrity_records",
    "memory_accuracy": "memory_accuracy_records",
    "memory_update": "memory_update_records",
}

# HaluMem's own real outcome vocabularies, from eval/evaluation.py's
# aggregate_eval_results() (the exact literal-string lists it checks membership
# against when deciding whether a record is "valid").
_QA_KNOWN_RESULT_TYPES = {"Correct", "Hallucination", "Omission"}
_UPDATE_KNOWN_TYPES = {"Correct", "Hallucination", "Omission", "Other"}


def load_eval_results(path: str) -> Dict[str, Any]:
    """Load a real HaluMem ``eval/evaluation.py`` output file (e.g.
    ``results/<frame>-<version>/<frame>_eval_stat_result.json``) from disk.

    That file is a single JSON object (``json.dump(eval_results, f, ...)`` in
    ``evaluation.py``'s ``main()``) -- not JSONL -- with keys
    ``overall_score``, ``memory_integrity_records``, ``memory_accuracy_records``,
    ``memory_update_records``, ``question_answering_records``. A thin convenience
    wrapper; every function in this module also accepts an already-loaded dict (or
    a plain ``list`` of that operation's records) directly, so this is optional.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_id(operation: str, uuid_: Any, ssession_id: Any, content: Any) -> str:
    """Deterministic test-case / result ID: a SHA-256 digest over HaluMem's own
    identifiers, per MemTensor/HaluMem#12 point 4. NOT Python's built-in `hash()`,
    which is per-process-randomized for strings and therefore unusable for an ID
    that has to mean the same thing on the next run or in a different process.
    """
    raw = "\x1f".join(str(x) for x in (operation, uuid_, ssession_id, content))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"halumem_{operation}_{uuid_}_s{ssession_id}_{digest}"


def _resolve_judge_model(judge_model: Optional[str]) -> str:
    """Per MemTensor/HaluMem#12 point 1: the judge model is configurable, read from
    the real `OPENAI_MODEL` environment variable `eval/llms.py` uses
    (`MODEL = os.getenv('OPENAI_MODEL')`) when not given explicitly. There is no
    hardcoded default -- an adapter that silently defaulted to some placeholder model
    name would misrepresent which model actually produced the verdicts being carried
    through.
    """
    if judge_model:
        return judge_model
    env_model = os.environ.get("OPENAI_MODEL")
    if env_model:
        return env_model
    raise ValueError(
        "judge_model was not given and the OPENAI_MODEL environment variable is not "
        "set. HaluMem's own evaluator reads its judge model from OPENAI_MODEL "
        "(eval/llms.py) rather than a hardcoded default; pass judge_model=... "
        "explicitly, or set OPENAI_MODEL, so the grader definition records the model "
        "that actually produced these verdicts."
    )


def _records_for(data: Union[Dict[str, Any], Sequence[Dict[str, Any]]], operation: str) -> List[Dict[str, Any]]:
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation {operation!r}; must be one of {OPERATIONS}")
    if isinstance(data, dict):
        return list(data.get(_RECORD_KEY[operation], []) or [])
    return list(data)


def _as_bool(value: Any) -> Optional[bool]:
    """HaluMem's `is_included_in_golden_memories` is documented and observed as the
    *strings* "true"/"false" (see eval/eval_tools.py's required output format), but
    evaluation.py's own aggregation checks `in ["true", "True"]` -- i.e. it treats the
    field defensively, not as a guaranteed-lowercase string. Mirror that leniency here
    rather than assuming a single casing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value in ("true", "True"):
            return True
        if value in ("false", "False"):
            return False
    return None


# ---------------------------------------------------------------------------
# Grader definitions (one llm_judge grader per operation)
# ---------------------------------------------------------------------------

# Adapted from eval/eval_tools.py's EVALUATION_PROMPT_FOR_QUESTION.
_QA_PROMPT = (
    "You are evaluating an AI memory system's answer to a question, using only the "
    "question, the reference answer, and the key memory points needed to derive it "
    "(no external knowledge). Classify the response as Correct, Hallucination, or "
    "Omission.\n\n"
    "Question: {input}\n"
    "Reference answer: {expected}\n"
    "Memory system response: {output}\n\n"
    "Correct: semantically equivalent to the reference answer, no contradictions with "
    "the key memory points.\n"
    "Hallucination: contradicts or is inconsistent with the reference answer or key "
    "memory points, or states a specific fact where the reference answer is "
    "unknown/uncertain.\n"
    "Omission: incomplete versus the reference answer, or explicitly disclaims "
    "memory despite relevant information existing in the key memory points.\n"
    'Return JSON: {"evaluation_result": "Correct|Hallucination|Omission", "reasoning": "..."}'
)

# Adapted from eval/eval_tools.py's EVALUATION_PROMPT_FOR_MEMORY_INTEGRITY.
_MEMORY_INTEGRITY_PROMPT = (
    "You are rating how well an AI memory system's extracted memories cover one "
    "expected memory point, on a scale of 0 (not mentioned or incorrect) to 2 (fully "
    "covered or implied); 1 means partially covered with key information missing, "
    "inaccurate, or slightly incorrect. Semantic matching is acceptable; exact "
    "wording is not required.\n\n"
    "Expected memory point: {expected}\n"
    "Context under test (identifies which point/session this is): {input}\n"
    "System's extracted memories: {output}\n\n"
    'Return JSON: {"score": "2|1|0", "reasoning": "..."}'
)

# Adapted from eval/eval_tools.py's EVALUATION_PROMPT_FOR_MEMORY_ACCURACY.
_MEMORY_ACCURACY_PROMPT = (
    "You are scoring the accuracy of one memory extracted by an AI memory system, "
    "using only the dialogue and golden (target) memory points -- no external "
    "knowledge. Score 2 if every information point in the candidate memory is "
    "supported with no contradictions or hallucinations; 1 if partially correct but "
    "also includes unsupported or contradictory content; 0 if entirely unsupported "
    "or contradictory (a hallucinated memory).\n\n"
    "Candidate memory under test: {input}\n"
    "Golden (target) memory points: {expected}\n"
    "Dialogue / extraction context: {output}\n\n"
    'Return JSON: {"accuracy_score": "2|1|0", "is_included_in_golden_memories": "true|false", "reason": "..."}'
)

# Adapted from eval/eval_tools.py's EVALUATION_PROMPT_FOR_UPDATE_MEMORY.
_MEMORY_UPDATE_PROMPT = (
    "You are evaluating whether an AI memory system's generated memories correctly "
    "include a target updated memory point, given the original (pre-update) memory. "
    "Classify as Correct (all information points from the target update are present "
    "and accurate, key fields like dates/values/proper nouns match exactly, the "
    "original is effectively replaced), Hallucination (a related new memory exists "
    "but contains factual errors or contradictions), Omission (no related new memory "
    "was generated, or one was generated but is missing key information), or Other "
    "(an update failure that doesn't clearly fit Hallucination or Omission).\n\n"
    "Target updated memory: {expected}\n"
    "Original memory (pre-update): {input}\n"
    "System's generated memories: {output}\n\n"
    'Return JSON: {"evaluation_result": "Correct|Hallucination|Omission|Other", "reason": "..."}'
)

_GRADER_PROMPTS = {
    "qa": _QA_PROMPT,
    "memory_integrity": _MEMORY_INTEGRITY_PROMPT,
    "memory_accuracy": _MEMORY_ACCURACY_PROMPT,
    "memory_update": _MEMORY_UPDATE_PROMPT,
}

_GRADER_DESCRIPTIONS = {
    "qa": "HaluMem question-answering hallucination/omission judge (llm_judge; carries HaluMem's own result_type verdict, does not re-grade).",
    "memory_integrity": "HaluMem memory-integrity (recall) judge (llm_judge; carries HaluMem's own memory_integrity_score verdict, does not re-grade).",
    "memory_accuracy": "HaluMem memory-accuracy (hallucination-in-extraction) judge (llm_judge; carries HaluMem's own memory_accuracy_score verdict, does not re-grade).",
    "memory_update": "HaluMem memory-update judge (llm_judge; carries HaluMem's own memory_update_type verdict, does not re-grade).",
}


def _grader_id(operation: str) -> str:
    return f"halumem_{operation}_judge"


def _make_grader(operation: str, judge_model: str) -> Dict[str, Any]:
    return {
        "id": _grader_id(operation),
        "type": "llm_judge",
        "description": _GRADER_DESCRIPTIONS[operation],
        "params": {"model": judge_model, "prompt": _GRADER_PROMPTS[operation]},
    }


# ---------------------------------------------------------------------------
# to_openeval — build an EvalSuite (TestCases) from HaluMem records
# ---------------------------------------------------------------------------


def _test_case_qa(idx: int, record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    question = record.get("question", "")
    tc_id = _stable_id("qa", uuid_, ssession_id, question)

    evidence = record.get("evidence") or []
    context = [e.get("memory_content", "") for e in evidence if isinstance(e, dict) and e.get("memory_content")]

    metadata: Dict[str, Any] = {
        "halumem.operation": "qa",
        "halumem.uuid": uuid_,
        "halumem.session_id": ssession_id,
        "halumem.evidence": evidence,
    }
    for key in ("question_type", "difficulty"):
        if key in record:
            metadata[f"halumem.{key}"] = record[key]

    tc: Dict[str, Any] = {
        "id": tc_id,
        "input": question,
        "graders": [_grader_id("qa")],
        "metadata": metadata,
    }
    if "answer" in record:
        tc["expected_output"] = record["answer"]
    if context:
        tc["context"] = context
    return tc


def _test_case_memory_integrity(idx: int, record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    content = record.get("memory_content", "")
    tc_id = _stable_id("memory_integrity", uuid_, ssession_id, content)

    metadata: Dict[str, Any] = {
        "halumem.operation": "memory_integrity",
        "halumem.uuid": uuid_,
        "halumem.session_id": ssession_id,
    }
    for key in ("memory_type", "importance", "memory_source", "is_update", "original_memories"):
        if key in record:
            metadata[f"halumem.{key}"] = record[key]

    tc: Dict[str, Any] = {
        # No separate dialogue/context is persisted on a real memory_integrity
        # record (see module docstring) -- the expected memory point itself, a
        # real HaluMem field, is the input under test.
        "id": tc_id,
        "input": content,
        "expected_output": content,
        "graders": [_grader_id("memory_integrity")],
        "metadata": metadata,
    }
    return tc


def _test_case_memory_accuracy(idx: int, record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    content = record.get("memory_content", "")
    tc_id = _stable_id("memory_accuracy", uuid_, ssession_id, content)

    metadata: Dict[str, Any] = {
        "halumem.operation": "memory_accuracy",
        "halumem.uuid": uuid_,
        "halumem.session_id": ssession_id,
    }

    tc: Dict[str, Any] = {
        # Same real-data limitation as memory_integrity: the dialogue and golden
        # memories used to grade this candidate aren't persisted on the record.
        "id": tc_id,
        "input": content,
        "graders": [_grader_id("memory_accuracy")],
        "metadata": metadata,
    }
    return tc


def _test_case_memory_update(idx: int, record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    content = record.get("memory_content", "")
    tc_id = _stable_id("memory_update", uuid_, ssession_id, content)

    metadata: Dict[str, Any] = {
        "halumem.operation": "memory_update",
        "halumem.uuid": uuid_,
        "halumem.session_id": ssession_id,
    }
    for key in ("memory_type", "importance", "memory_source", "is_update"):
        if key in record:
            metadata[f"halumem.{key}"] = record[key]

    original_memories = record.get("original_memories")

    tc: Dict[str, Any] = {
        "id": tc_id,
        "input": "\n".join(original_memories) if original_memories else "",
        "expected_output": content,
        "graders": [_grader_id("memory_update")],
        "metadata": metadata,
    }
    if original_memories:
        tc["context"] = list(original_memories)
    return tc


_TEST_CASE_BUILDERS = {
    "qa": _test_case_qa,
    "memory_integrity": _test_case_memory_integrity,
    "memory_accuracy": _test_case_memory_accuracy,
    "memory_update": _test_case_memory_update,
}


def to_openeval(
    data: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    operation: str,
    suite_id: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert HaluMem records for one operation into an EvalPort suite (dict).

    ``data`` is either a full HaluMem ``eval_results`` dict (as loaded by
    ``load_eval_results()``/produced by ``eval/evaluation.py``) -- the right record
    list is pulled out by ``operation`` -- or an already-extracted
    ``list[dict]`` of that operation's records directly.

    ``operation`` is one of ``OPERATIONS`` (``"qa"``, ``"memory_integrity"``,
    ``"memory_accuracy"``, ``"memory_update"``).

    ``judge_model`` names the LLM that produced (or will produce) the verdicts; if
    omitted, it is read from the ``OPENAI_MODEL`` environment variable, matching
    HaluMem's own ``eval/llms.py`` (see MemTensor/HaluMem#12 point 1). Raises
    ``ValueError`` if neither is available -- this module never fabricates a default
    model name.

    Returns a plain dict conforming to the EvalPort EvalSuite schema; pass it to
    ``openeval.validate.validate_suite()`` to confirm compliance.
    """
    records = _records_for(data, operation)
    resolved_model = _resolve_judge_model(judge_model)
    builder = _TEST_CASE_BUILDERS[operation]
    test_cases = [builder(i, r) for i, r in enumerate(records)]
    grader = _make_grader(operation, resolved_model)

    return {
        "version": _SPEC_VERSION,
        "id": suite_id or f"halumem_{operation}",
        "description": (
            f"MemTensor/HaluMem (https://github.com/MemTensor/HaluMem) "
            f"'{operation}' operation-level records, converted per "
            f"MemTensor/HaluMem#12."
        ),
        "test_cases": test_cases,
        "graders": [grader],
    }


# ---------------------------------------------------------------------------
# result_to_openeval — build a ResultSet from scored HaluMem records
# ---------------------------------------------------------------------------


def _result_qa(record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    question = record.get("question", "")
    tc_id = _stable_id("qa", uuid_, ssession_id, question)

    result_type = record.get("result_type")
    recognized = result_type in _QA_KNOWN_RESULT_TYPES

    metadata: Dict[str, Any] = {
        "halumem.result_type": result_type,  # canonical field: source of truth, never collapsed
    }
    if not recognized:
        metadata["halumem.unrecognized_result_type"] = True
    if "search_duration_ms" in record:
        metadata["halumem.search_duration_ms"] = record["search_duration_ms"]

    if recognized:
        passed = result_type == "Correct"
        score = 1.0 if passed else 0.0
    else:
        # Mirrors evaluation.py's own handling: a None/unrecognized result_type marks
        # the record is_valid=False and excludes it from the aggregate ratios rather
        # than silently scoring it as a hallucination.
        passed = False
        score = None

    grader_result: Dict[str, Any] = {
        "grader_id": _grader_id("qa"),
        "type": "llm_judge",
        "score": score,
        "passed": passed,
        "reason": str(result_type) if result_type is not None else "",
        "metadata": metadata,
    }

    return {
        "test_case_id": tc_id,
        "actual_output": record.get("system_response", ""),
        "grader_results": [grader_result],
        "passed": passed,
        "metadata": {},
    }


def _result_memory_integrity(record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    content = record.get("memory_content", "")
    tc_id = _stable_id("memory_integrity", uuid_, ssession_id, content)

    raw_score = record.get("memory_integrity_score")
    is_interference = record.get("memory_source") == "interference"

    metadata: Dict[str, Any] = {
        "halumem.memory_integrity_score": raw_score,  # canonical, verbatim (point 3)
        "halumem.memory_source": record.get("memory_source"),
    }
    for key in ("importance", "memory_type"):
        if key in record:
            metadata[f"halumem.{key}"] = record[key]

    if raw_score is None:
        passed = False
        normalized_score = None
    else:
        # Native scale is 0/1/2 (spec Validation Rule 5: normalize to [0,1], keep the
        # raw value in metadata.openeval.raw_score for exact reproducibility).
        normalized_score = raw_score / 2.0
        metadata["openeval.raw_score"] = raw_score
        # Interference (distractor) memories are the one case where NOT recalling is
        # correct -- aggregate_eval_results() counts interference_memory_scores when
        # memory_integrity_score == 0, the opposite of the == 2 check for real golden
        # memory points. Mirror that asymmetry here rather than a single passed rule.
        passed = (raw_score == 0) if is_interference else (raw_score == 2)

    grader_result: Dict[str, Any] = {
        "grader_id": _grader_id("memory_integrity"),
        "type": "llm_judge",
        "score": normalized_score,
        "passed": passed,
        "reason": f"HaluMem memory_integrity_score={raw_score}",
        "metadata": metadata,
    }

    return {
        "test_case_id": tc_id,
        # Not persisted on real memory_integrity records (see module docstring) --
        # left unset rather than fabricated.
        "grader_results": [grader_result],
        "passed": passed,
        "metadata": {},
    }


def _result_memory_accuracy(record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    content = record.get("memory_content", "")
    tc_id = _stable_id("memory_accuracy", uuid_, ssession_id, content)

    raw_score = record.get("memory_accuracy_score")
    included_raw = record.get("is_included_in_golden_memories")
    included_bool = _as_bool(included_raw)

    metadata: Dict[str, Any] = {
        "halumem.memory_accuracy_score": raw_score,  # canonical, verbatim (point 3)
        "halumem.is_included_in_golden_memories": included_raw,  # literal string, verbatim
    }

    if raw_score is None:
        passed = False
        normalized_score = None
    else:
        normalized_score = raw_score / 2.0
        metadata["openeval.raw_score"] = raw_score
        passed = raw_score == 2

    grader_result: Dict[str, Any] = {
        "grader_id": _grader_id("memory_accuracy"),
        "type": "llm_judge",
        "score": normalized_score,
        "passed": passed,
        "reason": (
            f"HaluMem memory_accuracy_score={raw_score}, "
            f"is_included_in_golden_memories={included_raw}"
        ),
        "metadata": metadata,
    }

    return {
        "test_case_id": tc_id,
        "grader_results": [grader_result],
        "passed": passed,
        "metadata": {"halumem.is_included_in_golden_memories_bool": included_bool},
    }


def _result_memory_update(record: Dict[str, Any]) -> Dict[str, Any]:
    uuid_ = record.get("uuid")
    ssession_id = record.get("ssession_id")
    content = record.get("memory_content", "")
    tc_id = _stable_id("memory_update", uuid_, ssession_id, content)

    update_type = record.get("memory_update_type")
    recognized = update_type in _UPDATE_KNOWN_TYPES

    metadata: Dict[str, Any] = {
        "halumem.memory_update_type": update_type,  # canonical field (point 2: Other stays distinct)
    }
    if not recognized:
        metadata["halumem.unrecognized_memory_update_type"] = True

    if recognized:
        passed = update_type == "Correct"
        score = 1.0 if passed else 0.0
    else:
        passed = False
        score = None

    grader_result: Dict[str, Any] = {
        "grader_id": _grader_id("memory_update"),
        "type": "llm_judge",
        "score": score,
        "passed": passed,
        "reason": str(update_type) if update_type is not None else "",
        "metadata": metadata,
    }

    memories_from_system = record.get("memories_from_system")

    result: Dict[str, Any] = {
        "test_case_id": tc_id,
        "grader_results": [grader_result],
        "passed": passed,
        "metadata": {},
    }
    if memories_from_system:
        result["actual_output"] = "\n".join(memories_from_system)
    return result


_RESULT_BUILDERS = {
    "qa": _result_qa,
    "memory_integrity": _result_memory_integrity,
    "memory_accuracy": _result_memory_accuracy,
    "memory_update": _result_memory_update,
}


def result_to_openeval(
    data: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    operation: str,
    suite_id: str,
    run_id: str,
    judge_model: Optional[str] = None,
    started_at: str = "1970-01-01T00:00:00Z",
) -> Dict[str, Any]:
    """Convert scored HaluMem records for one operation into an EvalPort ResultSet
    (dict). Carries HaluMem's own already-computed verdict through -- does not
    re-grade anything with an LLM call of its own.

    ``data``, ``operation`` and ``judge_model`` are as in ``to_openeval()``.
    ``started_at`` defaults to the Unix epoch since HaluMem's real eval_results
    output doesn't record a run start time; pass a real ISO-8601 timestamp for your
    own run if you have one.

    Returns a plain dict conforming to the EvalPort ResultSet schema; pass it to
    ``openeval.validate.validate_result_set()`` to confirm compliance.
    """
    records = _records_for(data, operation)
    resolved_model = _resolve_judge_model(judge_model)
    builder = _RESULT_BUILDERS[operation]
    results = [builder(r) for r in records]

    return {
        "version": _SPEC_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "provider": {"model": resolved_model},
        "runner": {"name": "halumem-openeval-adapter"},
        "results": results,
    }


# ---------------------------------------------------------------------------
# from_openeval — best-effort round trip back to HaluMem record shape
# ---------------------------------------------------------------------------

# metadata.halumem.* keys that came from a HaluMem field of the SAME name --
# i.e. a straight strip-prefix-and-copy-back round trip is correct for these.
# (operation/uuid/session_id are handled separately since they don't map back
# onto the plain record shape the same way.)
_DIRECT_METADATA_FIELDS = {
    "qa": ("question_type", "difficulty", "evidence"),
    "memory_integrity": ("memory_type", "importance", "memory_source", "is_update", "original_memories"),
    "memory_accuracy": (),
    "memory_update": ("memory_type", "importance", "memory_source", "is_update"),
}

# Which TestCase field holds the real HaluMem memory_content for each operation.
# memory_integrity/memory_accuracy set BOTH input and (for integrity) expected_output
# to memory_content, so either works -- but memory_update's TestCase.input is the
# *original* (pre-update) memories joined as text, with memory_content living only
# in expected_output, so it needs its own case rather than sharing "input" with the
# other two operations.
_CONTENT_TC_FIELD = {
    "memory_integrity": "input",
    "memory_accuracy": "input",
    "memory_update": "expected_output",
}


def from_openeval(suite: Dict[str, Any], operation: str) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite produced by ``to_openeval(..., operation)`` back
    into HaluMem record shape (best-effort round trip).

    Any ``TestCase.metadata["halumem.*"]`` key set by ``to_openeval()`` is unpacked
    back to its original HaluMem field name. Test cases with no such metadata
    (suites not originally produced by this adapter) still convert -- you just get
    the identifying fields back, with no fabricated values for fields that were
    never there.
    """
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation {operation!r}; must be one of {OPERATIONS}")

    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []) or []:
        metadata = tc.get("metadata") or {}
        row: Dict[str, Any] = {}

        uuid_ = metadata.get("halumem.uuid")
        ssession_id = metadata.get("halumem.session_id")
        if uuid_ is not None:
            row["uuid"] = uuid_
        if ssession_id is not None:
            row["ssession_id"] = ssession_id

        if operation == "qa":
            row["question"] = tc.get("input", "")
            if "expected_output" in tc:
                row["answer"] = tc["expected_output"]
        else:
            row["memory_content"] = tc.get(_CONTENT_TC_FIELD[operation], "")
            if operation == "memory_update" and tc.get("context"):
                # original_memories was carried on TestCase.context, not metadata
                # (see _test_case_memory_update) -- unlike every other field this
                # function round-trips, which come from metadata.halumem.*.
                row["original_memories"] = list(tc["context"])

        for field in _DIRECT_METADATA_FIELDS[operation]:
            meta_key = f"halumem.{field}"
            if meta_key in metadata:
                row[field] = metadata[meta_key]

        rows.append(row)
    return rows
