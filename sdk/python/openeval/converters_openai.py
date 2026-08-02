"""OpenAI Evals to EvalPort converter."""
from __future__ import annotations
from typing import Dict, List
from .types import OPENEVAL_VERSION


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
        tc = {
            "id": test.get("id", f"tc_{i}"),
            "input": test.get("input", test.get("prompt", "")),
            "graders": grader_ids,
        }
        if "target" in test:
            tc["expected_output"] = str(test["target"])
        elif "ideal" in test:
            tc["expected_output"] = str(test["ideal"])
        if "context" in test:
            tc["context"] = test["context"] if isinstance(test["context"], list) else [test["context"]]
        if "metadata" in test:
            tc["metadata"] = test["metadata"]
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
