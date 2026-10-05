"""open-instruct judge output <-> EvalPort adapter.

Standalone, offline converter between judged examples produced by
allenai/open-instruct's judge machinery (``open_instruct/judge_utils.py``:
``JUDGE_PROMPT_MAP`` / ``EXTRACTOR_MAP``, ``extract_json_score_with_fallback()``,
``extract_score_web_instruct()``, ``extract_score_with_fallback_max_10()``) and
the EvalPort interchange format (https://github.com/adhabnr-ux/evalport).

Why this exists
----------------
``extract_json_score_with_fallback()`` plus the ``JUDGE_PROMPT_MAP`` /
``EXTRACTOR_MAP`` registry already produce a ``(reasoning, score)`` pair per
``{input, output, label}`` triple -- that's structurally an EvalPort
``GraderResult`` (score + rationale) attached to a ``Result``. This package
turns a batch of those judged examples into a real, schema-valid EvalPort
``ResultSet``. See the design discussion in
https://github.com/allenai/open-instruct/issues/1922, including the
maintainer's review of the requirements this module implements.

Why it does not ``import open_instruct``
-----------------------------------------
``open_instruct``'s own ``__init__.py`` pulls in vllm / deepspeed /
flash-attn / torch -- a full training+inference stack this adapter has no
reason to require just to reshape already-judged records after the fact.
Per the reviewer's note that "an offline adapter needs no judge calls or
training dependencies," ``_extract_score()`` below is a self-contained,
pure-stdlib reimplementation of the *exact* parsing behavior of
``extract_json_score_with_fallback()`` and ``extract_score_web_instruct()``
in ``open_instruct/judge_utils.py`` as of commit
``677512ff387914b083a444db2c7a05ca995c114c`` (2026-10-05) -- same
markdown-fence stripping, same backslash-escaping, same JSON-then-regex
fallback -- so this package can run in CI with nothing installed but
``evalport-sdk``. If that upstream parsing logic changes, this commit pin is
the signal to re-verify this file against it.

Score scale and normalization
------------------------------
Per the open-instruct ``EXTRACTOR_MAP``: ``quality`` and ``quality_ref`` are
divided by 10 (their raw judge scale is 1-10); ``quality_rubric``, ``safety``,
``factuality``, ``creative_writing`` and ``refusal`` use
``extract_json_score_with_fallback()`` directly, with no further scaling;
``web_instruct_general_verifier`` is its own binary (0.0/1.0) parser. This
adapter applies that same per-``judge_type`` rule exactly once. If a caller
passes an already-extracted ``score`` (e.g. one produced by calling the real
``EXTRACTOR_MAP[judge_type]`` themselves), that score is trusted as already
normalized and is *never* divided again.

Parse failures vs. a genuine zero
-----------------------------------
Unlike the upstream extractors -- which collapse an unparseable judge
response to a score of ``0.0`` -- this adapter does not. A parse failure
is represented as ``score: None`` / ``passed: False`` (EvalPort
SPEC.md Validation Rule 6: "``score: null`` means not verified"), distinct
from a genuine judged zero (``score: 0.0``, ``parse_status: "json"``). The
raw judge response is always preserved in
``grader_results[].metadata.raw_judge_response`` when supplied, so a parse
failure can still be inspected and nothing is silently thrown away.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = ["to_openeval", "from_openeval", "KNOWN_JUDGE_TYPES", "__version__"]
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Mirrors open_instruct/judge_utils.py's JUDGE_PROMPT_MAP / EXTRACTOR_MAP keys
# (commit 677512f). Keeping this list explicit -- rather than accepting any
# string -- means an unrecognized judge_type fails loudly instead of silently
# mis-normalizing a score.
# ---------------------------------------------------------------------------
_DIV10_JUDGE_TYPES = frozenset({"quality", "quality_ref"})
_JSON_JUDGE_TYPES = frozenset(
    {"quality", "quality_rubric", "quality_ref", "safety", "factuality", "creative_writing", "refusal"}
)
_WEB_VERIFIER_JUDGE_TYPE = "web_instruct_general_verifier"
KNOWN_JUDGE_TYPES = frozenset(_JSON_JUDGE_TYPES | {_WEB_VERIFIER_JUDGE_TYPE})


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a dict-like or attribute-like object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_json_score(raw: str) -> Tuple[Optional[str], Optional[float], str]:
    """Reimplementation of judge_utils.extract_json_score_with_fallback().

    Returns ``(reasoning, score, parse_status)`` where ``parse_status`` is
    one of ``"json"`` (clean JSON parse), ``"regex_fallback"`` (JSON parsing
    failed but a ``"SCORE": ...`` key was recovered by regex), or
    ``"failed"`` (neither worked). Unlike upstream, a ``"failed"`` parse
    returns ``score=None`` rather than silently defaulting to ``0.0``, so
    callers -- and the validator -- can tell the two apart.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    cleaned = cleaned.replace("\r\n", "\n").replace("\n", "\\n")
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        reasoning = data.get("REASONING", "")
        score = float(data.get("SCORE", 0.0))
        return reasoning, score, "json"
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        match = re.search(r'"SCORE"\s*:\s*"?([0-9]+(?:\.[0-9]+)?)"?', cleaned)
        if match:
            return cleaned, float(match.group(1)), "regex_fallback"
        return None, None, "failed"


def _extract_web_verifier_score(raw: str) -> Tuple[Optional[str], Optional[float], str]:
    """Reimplementation of judge_utils.extract_score_web_instruct()."""
    lowered = raw.lower()
    if "final decision: yes" in lowered:
        return raw, 1.0, "json"
    if "final decision: no" in lowered:
        return raw, 0.0, "json"
    return None, None, "failed"


def _extract_score(judge_type: str, raw: str) -> Tuple[Optional[str], Optional[float], str]:
    if judge_type == _WEB_VERIFIER_JUDGE_TYPE:
        reasoning, score, status = _extract_web_verifier_score(raw)
    else:
        reasoning, score, status = _extract_json_score(raw)
    if score is not None and judge_type in _DIV10_JUDGE_TYPES:
        score = score / 10.0
    return reasoning, score, status


def _clip01(score: Optional[float]) -> Tuple[Optional[float], bool]:
    """Clip to EvalPort's required grader score range of [0, 1] or null.

    open-instruct's own templates for safety/factuality/creative_writing/
    refusal are marked TODO/incomplete upstream (see judge_utils.py), so
    their real score range isn't contractually guaranteed to stay inside
    [0, 1] the way quality/quality_ref (post /10) and the binary judges
    are. Clip defensively rather than let validate_result_set() reject the
    whole batch over one out-of-range judge score, and flag it (via the
    returned bool) so it's visible rather than silent.
    """
    if score is None:
        return None, False
    if score < 0.0:
        return 0.0, True
    if score > 1.0:
        return 1.0, True
    return score, False


def _default_extractor_name(judge_type: str) -> str:
    if judge_type == _WEB_VERIFIER_JUDGE_TYPE:
        return "extract_score_web_instruct"
    if judge_type in _DIV10_JUDGE_TYPES:
        return "extract_score_with_fallback_max_10"
    return "extract_json_score_with_fallback"


def _case_id(example: Any, index: int) -> str:
    cid = _get(example, "id") or _get(example, "case_id")
    return str(cid) if cid else f"tc_{index}"


def _to_result(example: Any, index: int, pass_threshold: float) -> Dict[str, Any]:
    judge_type = _get(example, "judge_type")
    if judge_type not in KNOWN_JUDGE_TYPES:
        raise ValueError(
            f"example {index} ({_case_id(example, index)!r}): unknown judge_type {judge_type!r}; "
            f"expected one of {sorted(KNOWN_JUDGE_TYPES)} "
            "(open_instruct.judge_utils.JUDGE_PROMPT_MAP's keys)"
        )

    raw_response = _get(example, "raw_judge_response")
    pre_reasoning = _get(example, "reasoning")
    pre_score = _get(example, "score")

    if pre_score is not None:
        # Caller already ran the real open_instruct EXTRACTOR_MAP[judge_type]
        # (or an equivalent) -- trust it as already-normalized and do NOT
        # divide by 10 again, even for quality/quality_ref.
        reasoning = pre_reasoning
        score: Optional[float] = float(pre_score)
        parse_status = "precomputed"
    elif raw_response is not None:
        reasoning, score, parse_status = _extract_score(judge_type, raw_response)
        if pre_reasoning is not None:
            reasoning = pre_reasoning
    else:
        raise ValueError(
            f"example {index} ({_case_id(example, index)!r}): need either a precomputed "
            "'score' or a 'raw_judge_response' to extract one from"
        )

    score, clipped = _clip01(score)
    verified = parse_status != "failed"
    passed = bool(verified and score is not None and score >= pass_threshold)

    grader_metadata: Dict[str, Any] = {"judge_type": judge_type, "parse_status": parse_status}
    judge_model = _get(example, "judge_model")
    if judge_model:
        grader_metadata["judge_model"] = judge_model
    grader_metadata["prompt_key"] = _get(example, "prompt_key") or judge_type
    grader_metadata["extractor"] = _get(example, "extractor_name") or _default_extractor_name(judge_type)
    version = _get(example, "version") or _get(example, "commit")
    if version:
        grader_metadata["source_version"] = version
    if raw_response is not None:
        grader_metadata["raw_judge_response"] = raw_response
    if clipped:
        grader_metadata["score_clipped"] = True

    grader_result: Dict[str, Any] = {
        "grader_id": f"open_instruct_{judge_type}",
        "type": "llm_judge",
        "score": score,
        "passed": passed,
        "metadata": grader_metadata,
    }
    if reasoning:
        grader_result["reason"] = reasoning

    result_metadata: Dict[str, Any] = {"judge_type": judge_type}
    input_ = _get(example, "input")
    if input_ is not None:
        result_metadata["input"] = input_
    label = _get(example, "label")
    if label is not None:
        result_metadata["label"] = label
    extra_metadata = _get(example, "metadata")
    if extra_metadata:
        result_metadata.update(dict(extra_metadata))

    result: Dict[str, Any] = {
        "test_case_id": _case_id(example, index),
        "passed": passed,
        "grader_results": [grader_result],
        "metadata": result_metadata,
    }
    output = _get(example, "output")
    if output is not None:
        result["actual_output"] = str(output)
    return result


def to_openeval(
    judged_examples: Sequence[Any],
    suite_id: str,
    run_id: str,
    started_at: Optional[str] = None,
    pass_threshold: float = 0.5,
    provider: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a batch of open-instruct judged examples into an EvalPort ResultSet.

    Each item in ``judged_examples`` may be a dict or an attribute-bearing
    object exposing:

    - ``judge_type`` (required): one of ``open_instruct.judge_utils.JUDGE_PROMPT_MAP``'s
      keys -- ``"quality"``, ``"quality_rubric"``, ``"quality_ref"``, ``"safety"``,
      ``"factuality"``, ``"creative_writing"``, ``"refusal"``, or
      ``"web_instruct_general_verifier"``.
    - EITHER ``score`` (a float already produced by the real
      ``EXTRACTOR_MAP[judge_type]``, trusted as-is and never re-normalized),
      OR ``raw_judge_response`` (the judge model's raw string reply, which
      this adapter parses and normalizes itself, mirroring the real
      extractor for that ``judge_type``).
    - ``input``, ``output``, ``label``, ``id``/``case_id``, ``judge_model``,
      ``prompt_key``, ``extractor_name``, ``version``/``commit``,
      ``metadata``: all optional, preserved on the output ``Result`` for
      provenance and debugging. ``reasoning`` may also be passed explicitly
      to override/supply the rationale text.

    A case with neither a usable ``score`` nor a ``raw_judge_response`` that
    parses raises ``ValueError`` -- except an unparseable
    ``raw_judge_response``, which is a normal, representable outcome
    (``score: None``, ``passed: False``, ``parse_status: "failed"``), not an
    exception.

    Returns a plain dict conforming to EvalPort's ``ResultSet`` schema. Pass
    it to ``openeval.validate.validate_result_set()`` to confirm compliance.
    """
    if not judged_examples:
        raise ValueError("judged_examples must be non-empty (ResultSet.results is required and non-empty)")

    results = [_to_result(ex, i, pass_threshold) for i, ex in enumerate(judged_examples)]

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
        "metadata": {"openeval": {"source": "open_instruct_judge_utils"}},
    }
    if provider:
        result_set["provider"] = provider
    return result_set


def from_openeval(result_set: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct judged-example dicts from an EvalPort ResultSet.

    Inverse of ``to_openeval()`` for a ResultSet this adapter itself
    produced: each returned dict carries ``id``, ``judge_type``, ``score``,
    ``reason``, ``raw_judge_response`` (if present), ``parse_status``,
    provenance fields, ``input``/``output``/``label`` and ``passed``.
    """
    examples: List[Dict[str, Any]] = []
    for result in result_set.get("results", []):
        grader_results = result.get("grader_results") or [{}]
        gr = grader_results[0]
        gr_metadata = gr.get("metadata") or {}
        result_metadata = result.get("metadata") or {}
        examples.append(
            {
                "id": result.get("test_case_id"),
                "judge_type": gr_metadata.get("judge_type"),
                "score": gr.get("score"),
                "reason": gr.get("reason"),
                "raw_judge_response": gr_metadata.get("raw_judge_response"),
                "parse_status": gr_metadata.get("parse_status"),
                "judge_model": gr_metadata.get("judge_model"),
                "prompt_key": gr_metadata.get("prompt_key"),
                "extractor_name": gr_metadata.get("extractor"),
                "input": result_metadata.get("input"),
                "output": result.get("actual_output"),
                "label": result_metadata.get("label"),
                "passed": result.get("passed"),
            }
        )
    return examples
