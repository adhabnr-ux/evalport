"""OpenAI Evals to EvalPort converter.

``sample["input"]`` in a real openai/evals JSONL data file is not always a
plain string. Confirmed by reading the actual eval class source in
openai/evals (``evals/elsuite/basic/match.py``'s ``Match.eval_sample``,
which asserts ``"input" in sample`` and then passes ``sample["input"]``
straight to a completion function as a prompt): for chat-based evals
(the common case -- ``Match``, ``Includes``, ``FuzzyMatch``, and the
model-graded eval classes all support it), ``input`` is a list of
``{"role": ..., "content": ...}`` chat messages, not a string. Before this
fix, ``from_openai_evals()`` passed that list straight through as
``TestCase.input`` -- which is invalid: ``spec/schemas/testcase.json``
requires ``input`` to be a non-empty string or an array of non-empty
strings, not an array of message objects. Confirmed with a real
``validate_suite()`` call: a chat-shaped sample produced
``{'path': '$.test_cases[0].$.input', 'message': 'input required', 'code':
'REQUIRED'}`` -- the validator doesn't coerce an array-of-objects into
"missing input", it just rejects it outright.

Likewise, ``sample["ideal"]`` may be a list of acceptable answer strings,
not just a single string (same ``Match.eval_sample`` assertion:
``isinstance(sample["ideal"], str) or isinstance(sample["ideal"], list)``).
The previous code did ``str(test["ideal"])`` unconditionally, which for a
list produced a Python repr string like ``"['4', 'four']"`` as the
"expected output" -- technically schema-valid (it's a string) but not a
meaningful one.

Both are fixed below: chat-message-list input is flattened into a single
readable string (``"role: content"`` per message, newline-joined) with the
original message list preserved verbatim under
``metadata["openai_evals"]["messages"]`` for lossless reconstruction;
list-valued ``ideal`` uses its first entry as ``expected_output`` with the
full list preserved under ``metadata["openai_evals"]["ideal_variants"]``.
"""
from __future__ import annotations
from typing import Any, Dict, List, Union
from .types import OPENEVAL_VERSION


def _is_chat_message(item: Any) -> bool:
    return isinstance(item, dict) and ("role" in item or "content" in item)


def _flatten_input(raw_input: Any) -> tuple[Union[str, List[str]], Dict[str, Any]]:
    """Return (schema-valid TestCase.input, extra per-test-case metadata).

    Handles the three real shapes openai/evals' own eval classes accept for
    ``sample["input"]``: a plain string, a list of chat-message dicts, or
    (defensively) a list of plain strings.
    """
    if isinstance(raw_input, str):
        return (raw_input if raw_input else "(empty input)"), {}

    if isinstance(raw_input, list) and raw_input:
        if all(_is_chat_message(m) for m in raw_input):
            flattened = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in raw_input
            )
            return flattened, {"messages": raw_input}
        if all(isinstance(m, str) for m in raw_input):
            return list(raw_input), {}

    # Unrecognized/empty shape -- don't fabricate a prompt; preserve the raw
    # value in metadata and fall back to a placeholder string rather than
    # crash the whole conversion over one malformed sample.
    return "(unrecognized input shape)", {"raw_input": raw_input}


def _flatten_ideal(raw_ideal: Any) -> tuple[str, Dict[str, Any]]:
    if isinstance(raw_ideal, list) and raw_ideal:
        return str(raw_ideal[0]), {"ideal_variants": raw_ideal}
    return str(raw_ideal), {}


def from_openai_evals(data: Dict) -> Dict:
    eval_id = data.get("id", "openai_eval_import")
    test_data = data.get("test_data", [])
    eval_config = data.get("config", {})
    sampling = eval_config.get("sampling", {})
    grader_spec = eval_config.get("grader", {})

    graders: List[Dict] = []
    if grader_spec:
        graders.append(_openai_grader_to_openeval("gr_0", grader_spec))
    if not graders:
        graders = [{"id": "gr_0", "type": "exact_match"}]

    grader_ids = [g["id"] for g in graders]

    tcs: List[Dict] = []
    for i, test in enumerate(test_data):
        raw_input = test.get("input", test.get("prompt", ""))
        flat_input, input_meta = _flatten_input(raw_input)

        tc: Dict[str, Any] = {
            "id": test.get("id", f"tc_{i}"),
            "input": flat_input,
            "graders": grader_ids,
        }

        extra_meta: Dict[str, Any] = dict(input_meta)
        if "target" in test:
            tc["expected_output"] = str(test["target"])
        elif "ideal" in test:
            expected, ideal_meta = _flatten_ideal(test["ideal"])
            tc["expected_output"] = expected
            extra_meta.update(ideal_meta)
        if "context" in test:
            tc["context"] = test["context"] if isinstance(test["context"], list) else [test["context"]]

        metadata: Dict[str, Any] = {}
        if "metadata" in test:
            metadata.update(test["metadata"])
        if extra_meta:
            metadata["openai_evals"] = extra_meta
        if metadata:
            tc["metadata"] = metadata

        tcs.append(tc)

    config = {}
    if "model" in sampling:
        config = {"provider": {"model": sampling["model"]}}
    if "temperature" in sampling:
        config.setdefault("provider", {})["temperature"] = sampling["temperature"]

    return {
        "version": OPENEVAL_VERSION,
        "id": f"suite_{eval_id}",
        "name": f"Imported from OpenAI Evals: {eval_id}",
        "graders": graders,
        "test_cases": tcs,
        "config": config,
        "metadata": {"openeval": {"source": "openai_evals"}},
    }


def _openai_grader_to_openeval(gid: str, spec: Dict) -> Dict:
    gtype = spec.get("type", spec.get("name", ""))
    gtype_lower = gtype.lower() if isinstance(gtype, str) else str(gtype).lower()

    if "exact" in gtype_lower or "match" in gtype_lower:
        return {"id": gid, "type": "exact_match"}
    if "includes" in gtype_lower or "contains" in gtype_lower:
        return {"id": gid, "type": "contains", "params": {"substring": spec.get("substring", "")}}
    if "regex" in gtype_lower or "pattern" in gtype_lower:
        return {"id": gid, "type": "regex", "params": {"pattern": spec.get("pattern", ".*")}}
    if "json" in gtype_lower:
        return {"id": gid, "type": "json_schema", "params": {"schema": spec.get("schema", {"type": "object"})}}
    if "model_graded" in gtype_lower or "modelgraded" in gtype_lower or "llm" in gtype_lower:
        prompt = spec.get("prompt", spec.get("instructions", "Evaluate if {output} is correct. Return JSON: {\"score\": 0.0-1.0}"))
        return {
            "id": gid,
            "type": "model graded",
            "params": {
                "model": spec.get("model", "gpt-4o"),
                "prompt": prompt,
            },
        }
    return {"id": gid, "type": "custom", "params": {"handler": f"openai_evals:{gtype}"}}
