"""Inspect AI to EvalPort converter."""
from __future__ import annotations
from typing import Dict, List
from .types import OPENEVAL_VERSION


def from_inspect(data: Dict) -> Dict:
    task_name = data.get("task", "inspect_import")
    samples = data.get("samples", [])
    scorers = data.get("scorers", ["exact"])

    graders: List[Dict] = []
    for i, scorer in enumerate(scorers):
        gid = f"gr_{i}"
        graders.append(_inspect_scorer_to_grader(gid, scorer))

    if not graders:
        graders = [{"id": "gr_0", "type": "exact_match"}]

    grader_ids = [g["id"] for g in graders]

    tcs: List[Dict] = []
    for sample in samples:
        tc = {
            "id": sample.get("id", f"tc_{len(tcs)}"),
            "input": sample.get("input", ""),
            "graders": grader_ids,
        }
        if "target" in sample:
            tc["expected_output"] = str(sample["target"])
        if "context" in sample:
            tc["context"] = sample["context"] if isinstance(sample["context"], list) else [sample["context"]]
        if "metadata" in sample:
            tc["metadata"] = sample["metadata"]
        tcs.append(tc)

    config = {}
    if "model" in data:
        config = {"provider": {"model": data["model"]}}

    return {
        "version": OPENEVAL_VERSION,
        "id": f"suite_inspect_{task_name}",
        "name": f"Imported from Inspect AI: {task_name}",
        "graders": graders,
        "test_cases": tcs,
        "config": config,
        "metadata": {"openeval": {"source": "inspect_ai"}},
    }


def _inspect_scorer_to_grader(gid: str, scorer: str) -> Dict:
    s = scorer.lower() if isinstance(scorer, str) else str(scorer).lower()
    if "exact" in s:
        return {"id": gid, "type": "exact_match"}
    if "pattern" in s or "regex" in s:
        return {"id": gid, "type": "regex", "params": {"pattern": ".*"}}
    if "includes" in s or "contains" in s:
        return {"id": gid, "type": "contains", "params": {"substring": ""}}
    if "json" in s:
        return {"id": gid, "type": "json_schema", "params": {"schema": {"type": "object"}}}
    if "model_graded" in s or "modelgraded" in s or "llm" in s:
        return {
            "id": gid,
            "type": "llm_judge",
            "params": {
                "model": "gpt-4o",
                "prompt": "Evaluate if {output} is correct for {input}. Expected: {expected}. Return JSON: {\"score\": 0.0-1.0}",
            },
        }
    if "manual" in s or "human" in s:
        return {"id": gid, "type": "human", "params": {"instructions": "Review the output manually."}}
    return {"id": gid, "type": "custom", "params": {"handler": f"inspect:{scorer}"}}
