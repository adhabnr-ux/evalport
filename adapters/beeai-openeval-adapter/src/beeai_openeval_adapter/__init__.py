"""Convert BeeAI Framework (https://github.com/i-am-bee/beeai-framework)
evaluation datasets and agent run outputs to/from the EvalPort open evaluation
format.

Filed and built following i-am-bee/beeai-framework#1677, where the BeeAI
maintainer preferred a standalone package in EvalPort's own ``adapters/``
directory over an in-tree exporter, and asked that it be exercised against
``RequirementAgent`` specifically.

The BeeAI shapes this module reads were taken from the framework's own source
(``python/beeai_framework/`` at ``beeai-framework==0.1.84``), not guessed:

- **Dataset side.** ``beeai_framework.evaluation`` only ships judge-LLM
  adapters (``DeepEvalLLM``, ``InstructorRagasLLM``); there is no in-tree
  dataset/test-case model. The shape BeeAI users actually produce is the one
  in ``python/examples/evaluation/dataset.json`` (loaded by
  ``examples/evaluation/dataset.py:load_items()``): a list of dicts
  ``{"question", "expected_answer", "supporting_sentences",
  "expected_tool_calls", "supporting_titles"}``, which both
  ``examples/evaluation/deepeval/experiment.py`` and
  ``examples/evaluation/ragas/experiment.py`` feed to a ``RequirementAgent``.
  ``to_openeval()``/``from_openeval()`` convert that item shape to/from an
  EvalPort ``Suite``.

- **Result side.** ``BaseAgent.run()`` returns ``AgentOutput``
  (``beeai_framework/agents/base.py``), a pydantic model extending
  ``RunnableOutput`` (``beeai_framework/runnable.py``) with
  ``output: list[AnyMessage]``, ``context: dict``, ``output_structured``,
  and a ``last_message`` property; ``RequirementAgent.run()`` returns the
  subclass ``RequirementAgentOutput`` which adds
  ``state: RequirementAgentRunState`` (``answer``, ``result``, ``memory``,
  ``iteration``, ``steps``, ``usage``, ``cost``). Messages
  (``beeai_framework/backend/message.py``) expose ``.text`` and
  ``.to_plain()``; assistant messages expose ``.get_tool_calls()``.
  ``result_to_openeval()``/``results_to_openeval()`` convert one or many of
  those into an EvalPort ``ResultSet``.

Everything is duck-typed: pass real BeeAI objects, or plain dicts / simple
objects carrying the same attribute names, and ``beeai_framework`` is never
imported at module import time (the runner version lookup uses
``importlib.metadata`` and degrades to ``None``). Anything EvalPort has no
first-class field for is preserved losslessly under ``metadata.beeai``, and
every document this module emits is tagged ``metadata.openeval.source =
"beeai-framework"`` like the other adapters in this repo.

Grading: BeeAI itself does not grade -- its evaluation examples hand the
``AgentOutput`` to DeepEval or Ragas metrics. So ``result_to_openeval()``
either (a) carries caller-supplied grader results through verbatim (any
metric objects/dicts exposing name/score/passed), or (b) when given an
``expected_output`` and nothing else, applies the same strict
``exact_match`` both BeeAI examples use as their first metric
(``ExactMatchMetric(threshold=1.0)`` / ``ragas ExactMatch``), or (c) with
neither, emits a single ``score: null`` / ``passed: false`` grader result so
an unscored run is visibly unscored rather than silently "passed".
``Result.error`` is used only for genuine run failures (an exception raised
by ``agent.run()``), mapped onto EvalPort's closed enum from BeeAI's own
error hierarchy (``ChatModelError``/``BackendError`` -> ``provider_error``,
timeouts -> ``timeout``, everything else -> ``runner_error``).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "result_to_openeval",
    "results_to_openeval",
    "__version__",
]
__version__ = "0.1.0"

_RESERVED_METADATA_KEY = "beeai"
_SOURCE_TAG = {"openeval": {"source": "beeai-framework"}}

# The strict equality check both BeeAI evaluation examples apply first
# (deepeval ExactMatchMetric(threshold=1.0) / ragas ExactMatch), expressed as
# EvalPort's standard grader.
EXACT_MATCH_GRADER: Dict[str, Any] = {
    "id": "gr_beeai_exact_match",
    "type": "exact_match",
    "description": (
        "Strict equality between the agent's final answer and expected_answer, "
        "matching the ExactMatch metric BeeAI's evaluation examples run first."
    ),
}

# Emitted when a Result carries no grading information at all, so the
# document validates and the run reads as unscored rather than passed.
UNSCORED_GRADER_ID = "gr_beeai_unscored"

# Dataset item keys this module maps onto first-class EvalPort fields
# (examples/evaluation/dataset.json shape, plus the DeepEval Golden aliases
# examples/evaluation/deepeval/experiment.py::rag_goldens() maps them to).
_ITEM_ID_KEYS = ("id", "test_case_id", "name")
_ITEM_INPUT_KEYS = ("question", "input")
_ITEM_EXPECTED_KEYS = ("expected_answer", "expected_output", "answer")
_ITEM_CONTEXT_KEYS = ("supporting_sentences", "context")
_ITEM_CONSUMED_KEYS = frozenset(
    _ITEM_ID_KEYS
    + _ITEM_INPUT_KEYS
    + _ITEM_EXPECTED_KEYS
    + _ITEM_CONTEXT_KEYS
    + ("expected_tool_calls", "supporting_titles", "expected_tools", "tags")
)


# ---------------------------------------------------------------------------
# duck-typing helpers
# ---------------------------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attribute-like object."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _first(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value
    return default


def _json_safe(value: Any, _depth: int = 0) -> Any:
    """Coerce arbitrary BeeAI values (pydantic models, messages, datetimes,
    exceptions, ...) into JSON-serializable data without importing
    ``beeai_framework``. Anything unrecognized falls back to ``str()`` so a
    document is always ``json.dump``-able; nothing raises."""
    if _depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth + 1) for v in value]
    # beeai_framework.backend.message.Message
    to_plain = getattr(value, "to_plain", None)
    if callable(to_plain):
        try:
            return _json_safe(to_plain(), _depth + 1)
        except Exception:  # pragma: no cover - defensive
            pass
    # pydantic v2 models (AgentOutput, ChatModelUsage, MessageToolCallContent, ...)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json"), _depth + 1)
        except Exception:
            try:
                return _json_safe(model_dump(), _depth + 1)
            except Exception:  # pragma: no cover - defensive
                pass
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except Exception:  # pragma: no cover - defensive
            pass
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return str(value)


def _message_text(message: Any) -> str:
    """Text of a BeeAI message: ``Message.text`` on real objects, else the
    concatenated ``text`` content parts of a ``to_plain()``-shaped dict, else
    a bare ``"text"``/``"content"`` string."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    text = _get(message, "text")
    if isinstance(text, str):
        return text
    content = _get(message, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if _get(part, "type") in (None, "text"):
                part_text = _get(part, "text")
                if isinstance(part_text, str):
                    parts.append(part_text)
        return "".join(parts)
    return ""


def _tool_calls_from_messages(messages: Iterable[Any]) -> List[Dict[str, Any]]:
    """Collect ``tool-call`` content parts in observation order, the same way
    ``examples/evaluation/deepeval/experiment.py::extract_real_tool_calls``
    does -- from the assistant messages' own tool-call content, not from any
    self-reported summary."""
    calls: List[Dict[str, Any]] = []
    for message in messages:
        get_tool_calls = getattr(message, "get_tool_calls", None)
        if callable(get_tool_calls):
            try:
                parts: Iterable[Any] = get_tool_calls()
            except Exception:  # pragma: no cover - defensive
                parts = []
        else:
            content = _get(message, "content")
            parts = [p for p in content if _get(p, "type") == "tool-call"] if isinstance(content, list) else []
        for part in parts:
            tool_name = _get(part, "tool_name")
            if not tool_name:
                continue
            args = _get(part, "args")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    pass
            calls.append({"id": _get(part, "id"), "tool_name": str(tool_name), "args": _json_safe(args)})
    return calls


# RequirementAgent ends every run by calling its internal FinalAnswerTool
# (beeai_framework/agents/requirement/utils/_tool.py); BeeAI's own evaluation
# examples skip it when counting tool usage (count_tool_usage /
# extract_real_tool_calls in examples/evaluation/deepeval/experiment.py), so
# tools_called does too. The raw tool_calls list keeps it, losslessly.
_INTERNAL_TOOLS = frozenset({"final_answer"})


def _tools_called(tool_calls: Sequence[Dict[str, Any]]) -> List[str]:
    return [c["tool_name"] for c in tool_calls if c["tool_name"] not in _INTERNAL_TOOLS]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "grader"


def _runner_version() -> Optional[str]:
    """Installed ``beeai-framework`` version, without importing the package."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - py<3.8
        return None
    try:
        return version("beeai-framework")
    except PackageNotFoundError:
        return None
    except Exception:  # pragma: no cover - defensive
        return None


# ---------------------------------------------------------------------------
# dataset -> Suite
# ---------------------------------------------------------------------------


def to_openeval(
    items: Sequence[Any],
    *,
    suite_id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    expected_tool: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert BeeAI evaluation dataset items into an EvalPort ``Suite`` (dict).

    ``items`` is the list ``examples/evaluation/dataset.py:load_items()``
    returns -- dicts shaped ``{"question", "expected_answer",
    "supporting_sentences", "expected_tool_calls", "supporting_titles"}`` --
    or any dict/object exposing the DeepEval-``Golden`` aliases
    ``input``/``expected_output``/``context``. One ``TestCase`` per item:

    - ``question`` -> ``input``
    - ``expected_answer`` -> ``expected_output`` (a list of acceptable
      answers keeps its first entry there and the full list under
      ``metadata.beeai.acceptable_answers``, since EvalPort's field is a
      single string)
    - ``supporting_sentences`` -> ``context``
    - ``expected_tool_calls`` (an int) and ``supporting_titles`` ->
      ``metadata.beeai`` verbatim. The dataset does not name the tool, so
      ``expected_tools`` is only emitted when you pass ``expected_tool=``
      (e.g. ``"Wikipedia"``, the tool both BeeAI experiments hardcode), as
      ``[expected_tool] * expected_tool_calls``.
    - any other item key -> ``metadata.beeai.extra`` (lossless)

    Items without an ``id`` get ``tc_<index>``. Every test case references
    the shared ``gr_beeai_exact_match`` grader (see module docstring).

    Pass the result to ``openeval.validate.validate_suite()`` to confirm
    compliance, or ``json.dump()`` it to share as a ``.json`` suite file.
    """
    if not items:
        raise ValueError("items is empty -- nothing to convert")

    test_cases: List[Dict[str, Any]] = []
    for index, item in enumerate(items):
        question = _first(item, _ITEM_INPUT_KEYS)
        if not isinstance(question, str) or not question:
            raise ValueError(f"item {index} has no non-empty 'question'/'input': {item!r}")
        item_id = _first(item, _ITEM_ID_KEYS)
        beeai_meta: Dict[str, Any] = {}

        test_case: Dict[str, Any] = {
            "id": str(item_id) if item_id is not None else f"tc_{index}",
            "input": question,
            "graders": [EXACT_MATCH_GRADER["id"]],
        }

        expected = _first(item, _ITEM_EXPECTED_KEYS)
        if isinstance(expected, (list, tuple)):
            beeai_meta["acceptable_answers"] = [str(a) for a in expected]
            if expected:
                test_case["expected_output"] = str(expected[0])
        elif expected is not None:
            test_case["expected_output"] = str(expected)

        context = _first(item, _ITEM_CONTEXT_KEYS)
        if isinstance(context, (list, tuple)):
            test_case["context"] = [str(c) for c in context]
        elif isinstance(context, str):
            test_case["context"] = [context]

        expected_tool_calls = _get(item, "expected_tool_calls")
        if expected_tool_calls is not None:
            beeai_meta["expected_tool_calls"] = expected_tool_calls
        supporting_titles = _get(item, "supporting_titles")
        if supporting_titles is not None:
            beeai_meta["supporting_titles"] = [str(t) for t in supporting_titles]

        explicit_tools = _get(item, "expected_tools")
        if isinstance(explicit_tools, (list, tuple)) and explicit_tools:
            test_case["expected_tools"] = [
                str(_get(t, "name") or _get(t, "tool_name") or t) for t in explicit_tools
            ]
        elif expected_tool and isinstance(expected_tool_calls, int) and expected_tool_calls > 0:
            test_case["expected_tools"] = [expected_tool] * expected_tool_calls

        tags = _get(item, "tags")
        if isinstance(tags, (list, tuple)) and tags:
            test_case["tags"] = [str(t) for t in tags]

        if isinstance(item, dict):
            extra = {k: _json_safe(v) for k, v in item.items() if k not in _ITEM_CONSUMED_KEYS}
            if extra:
                beeai_meta["extra"] = extra

        metadata: Dict[str, Any] = {}
        if beeai_meta:
            metadata[_RESERVED_METADATA_KEY] = beeai_meta
        if metadata:
            test_case["metadata"] = metadata
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "id": suite_id or "beeai_suite",
        "graders": [dict(EXACT_MATCH_GRADER)],
        "test_cases": test_cases,
        "metadata": {
            **_SOURCE_TAG,
            _RESERVED_METADATA_KEY: {"source": "beeai-framework", "adapter_version": __version__},
        },
    }
    if name is not None:
        suite["name"] = name
    if description is not None:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reverse ``to_openeval()``: turn an EvalPort ``Suite`` into the list of
    dataset items ``examples/evaluation/dataset.py:load_items()`` returns, so
    any EvalPort suite can be run through a ``RequirementAgent`` with BeeAI's
    own evaluation examples unchanged.

    Every test case is converted (not only ones this module produced --
    importing foreign suites into BeeAI is the point of this direction). Each
    item carries ``id``, ``question``, ``expected_answer``,
    ``supporting_sentences``, ``expected_tool_calls`` and
    ``supporting_titles``; ``metadata.beeai`` values are restored when present
    and otherwise derived (``expected_tool_calls`` from ``len(expected_tools)``,
    ``supporting_titles`` empty). Extra keys stashed under
    ``metadata.beeai.extra`` are restored verbatim.
    """
    items: List[Dict[str, Any]] = []
    for test_case in suite.get("test_cases", []) or []:
        metadata = (test_case.get("metadata") or {}).get(_RESERVED_METADATA_KEY) or {}
        raw_input = test_case.get("input", "")
        question = raw_input if isinstance(raw_input, str) else "\n".join(str(t) for t in raw_input)

        expected: Union[str, List[str], None]
        if "acceptable_answers" in metadata:
            expected = list(metadata["acceptable_answers"])
        else:
            expected = test_case.get("expected_output")

        expected_tools = test_case.get("expected_tools") or []
        expected_tool_calls = metadata.get("expected_tool_calls", len(expected_tools) if expected_tools else 0)

        item: Dict[str, Any] = {
            "id": test_case.get("id"),
            "question": question,
            "expected_answer": expected,
            "supporting_sentences": list(test_case.get("context") or []),
            "expected_tool_calls": expected_tool_calls,
            "supporting_titles": list(metadata.get("supporting_titles") or []),
        }
        if expected_tools:
            item["expected_tools"] = list(expected_tools)
        if test_case.get("tags"):
            item["tags"] = list(test_case["tags"])
        extra = metadata.get("extra")
        if isinstance(extra, dict):
            for key, value in extra.items():
                item.setdefault(key, value)
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# AgentOutput -> Result / ResultSet
# ---------------------------------------------------------------------------

_ERROR_TYPES = ("timeout", "provider_error", "runner_error")


def _classify_error(error: Any) -> Tuple[str, str, Dict[str, Any]]:
    """Map a raised exception (or an already-shaped dict) onto EvalPort's
    closed ``error.type`` enum using BeeAI's own error hierarchy by class
    name (``beeai_framework.errors.FrameworkError`` subclasses), so no
    import is needed: ``ChatModelError``/``BackendError``/provider-ish names
    -> ``provider_error``; any ``*Timeout*`` -> ``timeout``; else
    ``runner_error``. Returns ``(type, message, beeai_metadata)``."""
    if isinstance(error, dict):
        err_type = error.get("type")
        if err_type not in _ERROR_TYPES:
            err_type = "runner_error"
        message = str(error.get("message", ""))
        dict_meta: Dict[str, Any] = {
            k: _json_safe(v) for k, v in error.items() if k not in ("type", "message", "code", "retryable")
        }
        return str(err_type), message, dict_meta

    mro_names = [cls.__name__ for cls in type(error).__mro__] if isinstance(error, BaseException) else []
    joined = " ".join(mro_names).lower()
    if "timeout" in joined:
        err_type = "timeout"
    elif any(n in mro_names for n in ("ChatModelError", "BackendError", "EmbeddingModelError")) or (
        "provider" in joined or "ratelimit" in joined or "apierror" in joined
    ):
        err_type = "provider_error"
    else:
        err_type = "runner_error"
    meta: Dict[str, Any] = {"error_class": type(error).__name__ if isinstance(error, BaseException) else str(type(error))}
    # beeai_framework.errors.FrameworkError carries an explain() with the full
    # cause chain; keep it when available.
    explain = getattr(error, "explain", None)
    if callable(explain):
        try:
            meta["explain"] = str(explain())
        except Exception:  # pragma: no cover - defensive
            pass
    return err_type, str(error), meta


def _normalize_grader_result(raw: Any, index: int) -> Dict[str, Any]:
    """Accept a GraderResult-shaped dict, or any metric-ish object/dict
    (DeepEval ``MetricData``: ``name``/``score``/``success``/``threshold``/
    ``reason``; Ragas ``MetricResult``: ``value``/``reason``; ...) and emit
    an EvalPort GraderResult. Extra fields are kept under ``metadata``.
    Scores outside [0, 1] are clamped and the raw value preserved."""
    grader_id = _get(raw, "grader_id")
    grader_name = _first(raw, ("metric_name", "name", "metric", "id"))
    if not grader_id:
        grader_id = f"gr_beeai_{_slug(str(grader_name))}" if grader_name is not None else f"gr_beeai_metric_{index}"
    grader_type = _get(raw, "type") or "custom"

    score = _first(raw, ("score", "value"))
    if isinstance(score, bool):
        score = 1.0 if score else 0.0
    raw_score = score
    if isinstance(score, (int, float)):
        score = float(score)
        if score < 0.0 or score > 1.0:
            score = min(1.0, max(0.0, score))
    elif score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None

    passed = _first(raw, ("passed", "success"))
    if passed is None:
        threshold = _get(raw, "threshold")
        if isinstance(score, float) and isinstance(threshold, (int, float)):
            passed = score >= float(threshold)
        else:
            passed = isinstance(score, float) and score >= 1.0
    passed = bool(passed)

    result: Dict[str, Any] = {
        "grader_id": str(grader_id),
        "type": str(grader_type),
        "score": score,
        "passed": passed,
    }
    reason = _get(raw, "reason")
    if isinstance(reason, str) and reason:
        result["reason"] = reason

    meta: Dict[str, Any] = {}
    if grader_name is not None and str(grader_name) != str(grader_id):
        meta["name"] = str(grader_name)
    if raw_score is not None and raw_score != score:
        meta["raw_score"] = _json_safe(raw_score)
    threshold = _get(raw, "threshold")
    if threshold is not None:
        meta["threshold"] = _json_safe(threshold)
    error = _get(raw, "error")
    if error:
        meta["error"] = _json_safe(error)
    if isinstance(raw, dict):
        consumed = {
            "grader_id", "type", "score", "value", "passed", "success", "reason",
            "metric_name", "name", "metric", "id", "threshold", "error", "metadata",
        }
        extra = {k: _json_safe(v) for k, v in raw.items() if k not in consumed}
        if extra:
            meta["extra"] = extra
        if isinstance(raw.get("metadata"), dict):
            meta.update(_json_safe(raw["metadata"]))
    if meta:
        result["metadata"] = meta
    return result


def _agent_output_metadata(output: Any) -> Dict[str, Any]:
    """Everything on an ``AgentOutput``/``RequirementAgentOutput`` EvalPort
    has no field for, JSON-safe, under one namespace."""
    meta: Dict[str, Any] = {}
    messages = _get(output, "output") or []
    if not isinstance(messages, (list, tuple)):
        messages = [messages]
    meta["output"] = [_json_safe(m) for m in messages]
    context = _get(output, "context")
    if context:
        meta["context"] = _json_safe(context)
    structured = _get(output, "output_structured")
    if structured is not None:
        meta["output_structured"] = _json_safe(structured)

    # RequirementAgentOutput.state (RequirementAgentRunState): the full
    # trajectory lives in state.memory.messages; usage/cost/iteration/steps
    # are the run's own accounting.
    state = _get(output, "state")
    if state is not None:
        state_meta: Dict[str, Any] = {}
        for key in ("iteration", "usage", "cost"):
            value = _get(state, key)
            if value is not None:
                state_meta[key] = _json_safe(value)
        steps = _get(state, "steps")
        if steps:
            state_meta["steps"] = [
                {
                    "id": _get(step, "id"),
                    "iteration": _get(step, "iteration"),
                    "tool": _get(_get(step, "tool"), "name"),
                    "input": _json_safe(_get(step, "input")),
                    "error": _json_safe(_get(step, "error")),
                }
                for step in steps
            ]
        memory = _get(state, "memory")
        trajectory = _get(memory, "messages") if memory is not None else None
        if trajectory is None and isinstance(memory, (list, tuple)):
            trajectory = memory
        if trajectory:
            state_meta["trajectory"] = [_json_safe(m) for m in trajectory]
            tool_calls = _tool_calls_from_messages(trajectory)
            if tool_calls:
                meta["tool_calls"] = tool_calls
                meta["tools_called"] = _tools_called(tool_calls)
        if state_meta:
            meta["state"] = state_meta

    if "tool_calls" not in meta:
        tool_calls = _tool_calls_from_messages(messages)
        if tool_calls:
            meta["tool_calls"] = tool_calls
            meta["tools_called"] = _tools_called(tool_calls)
    return meta


def _actual_output(output: Any) -> Optional[str]:
    """``AgentOutput.last_message.text`` (what both BeeAI experiments read as
    the answer), falling back to the JSON of ``output_structured`` when the
    final message carries no text."""
    if output is None:
        return None
    last = _get(output, "last_message")
    if last is None:
        messages = _get(output, "output") or []
        last = messages[-1] if isinstance(messages, (list, tuple)) and messages else None
    text = _message_text(last)
    if text:
        return text
    structured = _get(output, "output_structured")
    if structured is not None:
        safe = _json_safe(structured)
        return safe if isinstance(safe, str) else json.dumps(safe, ensure_ascii=False, sort_keys=True)
    return ""


def result_to_openeval(
    test_case_id: str,
    output: Any = None,
    *,
    expected_output: Optional[str] = None,
    grader_results: Optional[Sequence[Any]] = None,
    error: Any = None,
    duration_ms: Optional[Union[int, float]] = None,
    completed_at: Optional[str] = None,
    attempt: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert one BeeAI agent run into an EvalPort ``Result`` (dict).

    ``output`` is what ``await agent.run(...)`` returned: an ``AgentOutput``
    (``RequirementAgentOutput`` for ``RequirementAgent``), or a dict/object
    with the same attribute names. Mapping:

    - ``last_message.text`` -> ``actual_output`` (falls back to the JSON of
      ``output_structured`` when the last message has no text; ``None`` when
      there is no output at all, e.g. the run raised).
    - ``output`` (messages), ``context``, ``output_structured``, and for
      ``RequirementAgentOutput`` ``state.{iteration,usage,cost,steps}`` plus
      the full ``state.memory.messages`` trajectory -> ``metadata.beeai``,
      alongside the tool calls extracted from that trajectory
      (``metadata.beeai.tool_calls`` / ``tools_called``).
    - ``grader_results``: passed through when given (any metric-shaped
      dicts/objects, see ``_normalize_grader_result``); otherwise an
      ``exact_match`` against ``expected_output`` when that is given;
      otherwise one ``score: null`` / ``passed: false`` placeholder.
    - ``error``: an exception raised by ``agent.run()`` (or an EvalPort
      error dict) -> ``Result.error`` with ``type`` mapped from BeeAI's error
      hierarchy; the grader results are then unscored and ``passed`` is
      ``False``. Only pass this for genuine run failures.
    - ``duration_ms`` (rounded to int), ``completed_at``, ``attempt`` are
      carried through when given. ``metadata`` is merged into the result's
      metadata next to the ``beeai`` namespace.
    """
    if not test_case_id:
        raise ValueError("test_case_id is required")

    result: Dict[str, Any] = {"test_case_id": str(test_case_id)}
    beeai_meta: Dict[str, Any] = {}

    actual = _actual_output(output) if output is not None else None
    if actual is not None:
        result["actual_output"] = actual
    if output is not None:
        beeai_meta.update(_agent_output_metadata(output))

    graders: List[Dict[str, Any]]
    if error is not None:
        err_type, message, err_meta = _classify_error(error)
        err: Dict[str, Any] = {"type": err_type}
        if message:
            err["message"] = message
        if isinstance(error, dict):
            if "code" in error and isinstance(error["code"], (str, int)) and not isinstance(error["code"], bool):
                err["code"] = error["code"]
            if isinstance(error.get("retryable"), bool):
                err["retryable"] = error["retryable"]
        result["error"] = err
        if err_meta:
            beeai_meta["error"] = err_meta
        if grader_results:
            graders = [_normalize_grader_result(g, i) for i, g in enumerate(grader_results)]
        else:
            graders = [
                {
                    "grader_id": UNSCORED_GRADER_ID if expected_output is None else EXACT_MATCH_GRADER["id"],
                    "type": "custom" if expected_output is None else EXACT_MATCH_GRADER["type"],
                    "score": None,
                    "passed": False,
                    "reason": f"run failed ({err_type}); not graded",
                }
            ]
        result["passed"] = False
    elif grader_results:
        graders = [_normalize_grader_result(g, i) for i, g in enumerate(grader_results)]
        result["passed"] = all(g["passed"] for g in graders)
    elif expected_output is not None:
        matched = (actual or "").strip() == str(expected_output).strip()
        graders = [
            {
                "grader_id": EXACT_MATCH_GRADER["id"],
                "type": EXACT_MATCH_GRADER["type"],
                "score": 1.0 if matched else 0.0,
                "passed": matched,
            }
        ]
        result["passed"] = matched
    else:
        graders = [
            {
                "grader_id": UNSCORED_GRADER_ID,
                "type": "custom",
                "score": None,
                "passed": False,
                "reason": "no grader_results or expected_output supplied; unscored",
            }
        ]
        result["passed"] = False
    result["grader_results"] = graders

    if duration_ms is not None:
        result["duration_ms"] = max(0, int(round(float(duration_ms))))
    if completed_at is not None:
        result["completed_at"] = completed_at
    if attempt is not None:
        result["attempt"] = int(attempt)

    merged_meta: Dict[str, Any] = dict(metadata or {})
    if beeai_meta:
        merged_meta[_RESERVED_METADATA_KEY] = beeai_meta
    if merged_meta:
        result["metadata"] = merged_meta
    return result


def results_to_openeval(
    runs: Sequence[Any],
    *,
    suite_id: str,
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[Dict[str, Any]] = None,
    isolation: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert a batch of BeeAI agent runs into an EvalPort ``ResultSet``.

    Each entry of ``runs`` is either ``(test_case_id, agent_output)``, or a
    dict of ``result_to_openeval()`` keyword arguments (``test_case_id``,
    ``output``, ``expected_output``, ``grader_results``, ``error``,
    ``duration_ms``, ``completed_at``, ``attempt``, ``metadata``), or an
    already-built EvalPort ``Result`` dict (recognized by ``grader_results``
    being present), so you can mix ``result_to_openeval()`` output with raw
    runs.

    ``summary`` is computed from the results (``total``/``passed``/``failed``,
    ``pass_rate``, ``avg_score`` over non-null grader scores, ``duration_ms``
    when every result has one, and ``by_grader``). ``runner`` records
    ``beeai-framework`` and its installed version when the package is
    importable via ``importlib.metadata``. ``model`` becomes
    ``provider.model`` (e.g. ``ChatModel.model_id``); a full ``provider`` dict
    wins over ``model`` when both are given.
    """
    if not runs:
        raise ValueError("runs is empty -- nothing to convert")

    results: List[Dict[str, Any]] = []
    for index, run in enumerate(runs):
        if isinstance(run, dict) and "grader_results" in run and "test_case_id" in run:
            results.append(run)
        elif isinstance(run, dict):
            kwargs = dict(run)
            test_case_id = kwargs.pop("test_case_id", None)
            if test_case_id is None:
                raise ValueError(f"runs[{index}] has no 'test_case_id'")
            output = kwargs.pop("output", None)
            results.append(result_to_openeval(str(test_case_id), output, **kwargs))
        elif isinstance(run, (list, tuple)) and len(run) == 2:
            results.append(result_to_openeval(str(run[0]), run[1]))
        else:
            raise ValueError(
                f"runs[{index}] must be (test_case_id, output), a kwargs dict, or a Result dict; got {type(run).__name__}"
            )

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    scores = [
        g["score"]
        for r in results
        for g in r.get("grader_results", [])
        if isinstance(g.get("score"), (int, float)) and not isinstance(g.get("score"), bool)
    ]
    summary: Dict[str, Any] = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
    }
    if scores:
        summary["avg_score"] = min(1.0, max(0.0, sum(scores) / len(scores)))
    if results and all(isinstance(r.get("duration_ms"), int) for r in results):
        summary["duration_ms"] = sum(r["duration_ms"] for r in results)
    by_grader: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for g in r.get("grader_results", []):
            bucket = by_grader.setdefault(g["grader_id"], {"passed": 0, "failed": 0, "_scores": []})
            bucket["passed" if g.get("passed") else "failed"] += 1
            if isinstance(g.get("score"), (int, float)) and not isinstance(g.get("score"), bool):
                bucket["_scores"].append(float(g["score"]))
    for bucket in by_grader.values():
        bucket_scores = bucket.pop("_scores")
        if bucket_scores:
            bucket["avg_score"] = sum(bucket_scores) / len(bucket_scores)
    if by_grader:
        summary["by_grader"] = by_grader

    result_set: Dict[str, Any] = {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": summary,
        "metadata": {
            **_SOURCE_TAG,
            _RESERVED_METADATA_KEY: {"source": "beeai-framework", "adapter_version": __version__},
            **(metadata or {}),
        },
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    if isolation is not None:
        result_set["isolation"] = isolation
    if provider:
        result_set["provider"] = dict(provider)
    elif model:
        result_set["provider"] = {"model": model}
    runner: Dict[str, Any] = {"name": "beeai-framework"}
    version = _runner_version()
    if version:
        runner["version"] = version
    result_set["runner"] = runner
    return result_set
