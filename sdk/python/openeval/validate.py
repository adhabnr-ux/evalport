from __future__ import annotations
from typing import Any, List, Dict
from .types import ValidationResult
import re

STANDARD_GRADER_TYPES = {"exact_match","contains","regex","semantic_similarity","llm_judge","json_schema","json_path","code","human","model graded","custom"}

# Full semver 2.0.0 pattern (https://semver.org/#backusnaur-form-grammar-for-valid-semver-versions).
# Was previously hardcoded to accept only "X.Y.Z" or "X.Y.Z-draft" -- rejected legitimate
# prerelease versions like "1.0.0-rc.1" or "1.1.0-beta.2", which is what this project's own
# README and public communications already describe the spec version as.
SEMVER_RE = re.compile(
    r"^\d+\.\d+\.\d+"
    r"(?:-(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)

def _err(p,m,c): return {"path":p,"message":m,"code":c}

def validate_test_case(tc):
    errors=[]
    if not isinstance(tc,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(tc.get("id"),str) or not tc["id"]: errors.append(_err("$.id","id required","REQUIRED"))
    inp=tc.get("input")
    if not isinstance(inp,str) and not (isinstance(inp,list) and all(isinstance(x,str) for x in inp)): errors.append(_err("$.input","input required","REQUIRED"))
    elif isinstance(inp,list) and not inp: errors.append(_err("$.input","empty","MIN_ITEMS"))
    gr=tc.get("graders")
    if not isinstance(gr,list) or not gr: errors.append(_err("$.graders","graders required","REQUIRED"))
    else:
        for i,g in enumerate(gr):
            if isinstance(g,str):
                if not g: errors.append(_err(f"$.graders[{i}]","empty","EMPTY_STRING"))
            elif isinstance(g,dict):
                gv=validate_grader(g)
                if not gv.valid:
                    for e in gv.errors: errors.append(_err(f"$.graders[{i}].{e['path']}",e["message"],e["code"]))
            else: errors.append(_err(f"$.graders[{i}]","must be string or object","TYPE_ERROR"))
    return ValidationResult(not errors,errors)

def validate_grader(g):
    errors=[]
    if not isinstance(g,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(g.get("id"),str) or not g["id"]: errors.append(_err("$.id","id required","REQUIRED"))
    gt=g.get("type")
    if not isinstance(gt,str) or not gt: errors.append(_err("$.type","type required","REQUIRED"))
    else:
        p=g.get("params") or {}
        if gt in STANDARD_GRADER_TYPES:
            for e in _vp(gt,p): errors.append(_err(f"$.params.{e['path']}",e["message"],e["code"]))
        else:
            # Non-standard type name: valid, but treated exactly like "custom" -- must
            # identify a handler so a runner that doesn't recognize it can skip gracefully
            # instead of guessing. This is what lets an adapter use a descriptive type
            # (e.g. "trulens_feedback") instead of the generic "custom" bucket.
            for e in _vp("custom",p): errors.append(_err(f"$.params.{e['path']}",e["message"],e["code"]))
    return ValidationResult(not errors,errors)

def _vp(t,p):
    e=[]
    if t=="contains":
        if not isinstance(p.get("substring"),str) or not p["substring"]: e.append(_err("substring","required","REQUIRED"))
    elif t=="regex":
        if not isinstance(p.get("pattern"),str) or not p["pattern"]: e.append(_err("pattern","required","REQUIRED"))
    elif t=="semantic_similarity":
        th=p.get("threshold")
        if not isinstance(th,(int,float)) or th<0 or th>1: e.append(_err("threshold","0-1","OUT_OF_RANGE"))
    elif t=="llm_judge":
        if not isinstance(p.get("model"),str) or not p["model"]: e.append(_err("model","required","REQUIRED"))
        pr=p.get("prompt")
        if not isinstance(pr,str) or not pr: e.append(_err("prompt","required","REQUIRED"))
        elif "{output}" not in pr and "{input}" not in pr and "{expected}" not in pr: e.append(_err("prompt","missing token","MISSING_TOKEN"))
    elif t=="json_schema":
        if not isinstance(p.get("schema"),dict): e.append(_err("schema","required","REQUIRED"))
    elif t=="json_path":
        if not isinstance(p.get("path"),str) or not p["path"]: e.append(_err("path","required","REQUIRED"))
        if "expected" not in p: e.append(_err("expected","required","REQUIRED"))
    elif t=="code":
        if p.get("language") not in ("python","javascript"): e.append(_err("language","python|javascript","INVALID_VALUE"))
        if not isinstance(p.get("source"),str) or not p["source"]: e.append(_err("source","required","REQUIRED"))
    elif t=="custom":
        if not isinstance(p.get("handler"),str) or not p["handler"]: e.append(_err("handler","required","REQUIRED"))
    return e

def validate_suite(s):
    errors=[]
    if not isinstance(s,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(s.get("version"),str) or not SEMVER_RE.match(s.get("version","")): errors.append(_err("$.version","semver","INVALID_VERSION"))
    if not isinstance(s.get("id"),str) or not s["id"]: errors.append(_err("$.id","required","REQUIRED"))
    tcs=s.get("test_cases")
    if not isinstance(tcs,list) and not isinstance(s.get("test_cases_file"),str): errors.append(_err("$.test_cases","required","REQUIRED"))
    if isinstance(tcs,list):
        if not tcs: errors.append(_err("$.test_cases","empty","MIN_ITEMS"))
        ids=set()
        for i,tc in enumerate(tcs):
            tv=validate_test_case(tc)
            if not tv.valid:
                for e in tv.errors: errors.append(_err(f"$.test_cases[{i}].{e['path']}",e["message"],e["code"]))
            tid=tc.get("id") if isinstance(tc,dict) else None
            if isinstance(tid,str):
                if tid in ids: errors.append(_err(f"$.test_cases[{i}].id",f"dup:{tid}","DUPLICATE_ID"))
                ids.add(tid)
        grs=s.get("graders",[])
        if isinstance(grs,list):
            gids=set()
            for i,g in enumerate(grs):
                gv=validate_grader(g)
                if not gv.valid:
                    for e in gv.errors: errors.append(_err(f"$.graders[{i}].{e['path']}",e["message"],e["code"]))
                gid=g.get("id") if isinstance(g,dict) else None
                if isinstance(gid,str):
                    if gid in gids: errors.append(_err(f"$.graders[{i}].id",f"dup:{gid}","DUPLICATE_ID"))
                    gids.add(gid)
            for i,tc in enumerate(tcs):
                if isinstance(tc,dict) and isinstance(tc.get("graders"),list):
                    for j,gr in enumerate(tc["graders"]):
                        if isinstance(gr,str) and gr not in gids: errors.append(_err(f"$.test_cases[{i}].graders[{j}]",f"not found:{gr}","DANGLING_REFERENCE"))
    return ValidationResult(not errors,errors)

def validate_result_set(r):
    errors=[]
    if not isinstance(r,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(r.get("version"),str) or not SEMVER_RE.match(r.get("version","")): errors.append(_err("$.version","semver","INVALID_VERSION"))
    if not isinstance(r.get("suite_id"),str) or not r["suite_id"]: errors.append(_err("$.suite_id","required","REQUIRED"))
    run_id=r.get("run_id")
    if not isinstance(run_id,str) or not run_id: errors.append(_err("$.run_id","required","REQUIRED"))
    if not isinstance(r.get("started_at"),str): errors.append(_err("$.started_at","required","REQUIRED"))
    isolation=r.get("isolation")
    if isolation is not None and not isinstance(isolation,str): errors.append(_err("$.isolation","must be string","TYPE_ERROR"))
    rs=r.get("results")
    if not isinstance(rs,list) or not rs: errors.append(_err("$.results","required","REQUIRED"))
    else:
        # Discussion #22 / issue #20: (test_case_id, run_id, attempt) must be
        # unique across results whenever attempt is present -- the join key for
        # repeated trials of the same test case (LangSmith num_repetitions,
        # Promptfoo repeats, Inspect AI epochs). attempt is absent-by-default
        # and single-attempt-per-case ResultSets need no change; this check is a
        # no-op unless a producer actually opts into attempt.
        seen_attempts=set()
        for i,x in enumerate(rs):
            if not isinstance(x,dict): errors.append(_err(f"$.results[{i}]","object","TYPE_ERROR"));continue
            if not isinstance(x.get("test_case_id"),str): errors.append(_err(f"$.results[{i}].test_case_id","required","REQUIRED"))
            if not isinstance(x.get("passed"),bool): errors.append(_err(f"$.results[{i}].passed","required","REQUIRED"))
            if "attempt" in x and x["attempt"] is not None:
                attempt=x["attempt"]
                if not isinstance(attempt,int) or isinstance(attempt,bool) or attempt<1:
                    errors.append(_err(f"$.results[{i}].attempt","must be an integer >= 1","OUT_OF_RANGE"))
                else:
                    key=(x.get("test_case_id"),run_id,attempt)
                    if key in seen_attempts:
                        errors.append(_err(f"$.results[{i}].attempt",f"duplicate (test_case_id, run_id, attempt): {key}","DUPLICATE_ATTEMPT"))
                    else:
                        seen_attempts.add(key)
            grs=x.get("grader_results")
            if not isinstance(grs,list): errors.append(_err(f"$.results[{i}].grader_results","required","REQUIRED"))
            else:
                for j,gr in enumerate(grs):
                    if not isinstance(gr,dict): errors.append(_err(f"$.results[{i}].grader_results[{j}]","object","TYPE_ERROR"));continue
                    if not isinstance(gr.get("grader_id"),str): errors.append(_err(f"$.results[{i}].grader_results[{j}].grader_id","required","REQUIRED"))
                    if not isinstance(gr.get("type"),str): errors.append(_err(f"$.results[{i}].grader_results[{j}].type","required","REQUIRED"))
                    sc=gr.get("score")
                    if not isinstance(sc,(int,float,type(None))) or isinstance(sc,bool): errors.append(_err(f"$.results[{i}].grader_results[{j}].score","number|null","TYPE_ERROR"))
                    elif sc is not None and (sc<0 or sc>1): errors.append(_err(f"$.results[{i}].grader_results[{j}].score","must be in [0,1] or null","OUT_OF_RANGE"))
                    if not isinstance(gr.get("passed"),bool): errors.append(_err(f"$.results[{i}].grader_results[{j}].passed","required","REQUIRED"))
    return ValidationResult(not errors,errors)

def validate_document(d,t):
    if t=="testcase": return validate_test_case(d)
    if t=="grader": return validate_grader(d)
    if t=="suite": return validate_suite(d)
    if t=="resultset": return validate_result_set(d)
    raise ValueError(f"Unknown type: {t}")
