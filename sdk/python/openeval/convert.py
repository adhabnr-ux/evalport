from .types import OPENEVAL_VERSION

def from_promptfoo(pf):
    tests = pf.get("tests", [])
    providers = pf.get("providers", [])
    graders = []
    tcs = []
    for i, t in enumerate(tests):
        v = t.get("vars", {})
        asserts = t.get("assert", [])
        tg = []
        for j, a in enumerate(asserts):
            gid = f"gr_{i}_{j}"
            at = a.get("type", "")
            if at == "equals": graders.append({"id": gid, "type": "exact_match"})
            elif at == "contains": graders.append({"id": gid, "type": "contains", "params": {"substring": str(a.get("value", ""))}})
            else: graders.append({"id": gid, "type": "custom", "params": {"handler": f"promptfoo:{at}"}})
            tg.append(gid)
        inp = v.get("query") or v.get("prompt") or str(v)
        tc = {"id": f"tc_{i}", "input": inp, "graders": tg if tg else ["gr_default"]}
        if "expected" in v: tc["expected_output"] = v["expected"]
        tcs.append(tc)
    if not graders: graders = [{"id": "gr_default", "type": "exact_match"}]
    cfg = {}
    if providers and isinstance(providers[0], dict):
        p = providers[0]
        cfg = {"provider": {k: v for k, v in [("model", p.get("model"))] if v is not None}}
    return {"version": OPENEVAL_VERSION, "id": "suite_promptfoo_import", "name": "Imported from Promptfoo", "graders": graders, "test_cases": tcs, "config": cfg}

def compute_summary(results):
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    scores = [gr["score"] for r in results for gr in r.get("grader_results", []) if gr.get("score") is not None]
    return {"total": total, "passed": passed, "failed": total - passed, "pass_rate": passed / total if total else 0, "avg_score": sum(scores) / len(scores) if scores else 0}

def create_result_set(suite, results, run_id, runner_name="openeval-sdk", runner_version="1.0.0"):
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    return {"version": OPENEVAL_VERSION, "suite_id": suite["id"], "suite_version": suite.get("version"), "run_id": run_id, "started_at": now, "completed_at": now, "provider": suite.get("config", {}).get("provider"), "runner": {"name": runner_name, "version": runner_version}, "results": results, "summary": compute_summary(results)}
