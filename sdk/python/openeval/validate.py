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

# RFC 3339 section 5.6 `date-time` (what resultset.json's `format: "date-time"` means):
#   full-date "T" full-time, where full-time = partial-time time-offset,
#   partial-time = HH ":" MM ":" SS [time-secfrac], time-offset = "Z" / ("+" / "-") HH ":" MM.
# Seconds and an offset are both mandatory: a date-only value ("2026-01-15") or an
# offset-less local time ("2026-01-15T10:30:00") is NOT an RFC 3339 date-time, even
# though both are valid ISO 8601. RFC 3339 5.6 NOTE: "T" and "Z" may alternatively be
# lower case. time-second allows 60 (leap second) per the RFC 3339 grammar.
# [0-9] rather than \d so non-ASCII digits never match. Must be used with fullmatch
# (the TypeScript twin anchors with ^...$, which in JS never matches before a trailing
# newline -- Python's `$` would, so it is deliberately not used here).
RFC3339_DATE_TIME_RE = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})[Tt]"
    r"(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)(?:\.[0-9]+)?"
    r"(?:[Zz]|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)

_DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)

def is_rfc3339_date_time(v: Any) -> bool:
    """True iff v is a string that is a valid RFC 3339 date-time (incl. calendar-valid day)."""
    if not isinstance(v, str): return False
    m = RFC3339_DATE_TIME_RE.fullmatch(v)
    if not m: return False
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not 1 <= month <= 12: return False
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    max_day = 29 if (month == 2 and leap) else _DAYS_IN_MONTH[month - 1]
    return 1 <= day <= max_day

def _err(p,m,c): return {"path":p,"message":m,"code":c}

# JSON Schema type predicates. bool is a subclass of int in Python, so every numeric
# predicate excludes it explicitly -- JSON Schema's "integer"/"number" never match
# true/false. "integer" also matches an integral float (5.0), exactly as JSON Schema
# 2020-12 does ("integer" = any number with a zero fractional part).
def _is_num(v): return isinstance(v,(int,float)) and not isinstance(v,bool)
def _is_int(v): return (isinstance(v,int) and not isinstance(v,bool)) or (isinstance(v,float) and v.is_integer())

# Optional-field type checks. Each spec entry is (key, kind[, min[, max]]) or
# (key, "enum", allowed_values). A field is checked only when its key is PRESENT --
# including an explicit null, which JSON Schema's `type` keyword rejects for every
# field below (none of them is declared nullable). Unknown keys are deliberately NOT
# policed here (see test_schema_consistency.py -- that's a separate spec question).
# Mirrors sdk/typescript/src/validate.ts's checkOptional() rule-for-rule, including
# error codes and messages.
def _check_optional(obj, base, fields, errors):
    for f in fields:
        key, kind = f[0], f[1]
        if key not in obj: continue
        v = obj[key]
        path = f"{base}.{key}"
        if kind == "string":
            if not isinstance(v,str): errors.append(_err(path,"must be string","TYPE_ERROR"))
        elif kind == "object":
            if not isinstance(v,dict): errors.append(_err(path,"must be object","TYPE_ERROR"))
        elif kind == "boolean":
            if not isinstance(v,bool): errors.append(_err(path,"must be boolean","TYPE_ERROR"))
        elif kind == "string_array":
            if not isinstance(v,list) or not all(isinstance(x,str) for x in v): errors.append(_err(path,"must be array of strings","TYPE_ERROR"))
        elif kind == "string_or_integer":
            if not isinstance(v,str) and not _is_int(v): errors.append(_err(path,"must be string or integer","TYPE_ERROR"))
        elif kind == "date_time":
            if not isinstance(v,str): errors.append(_err(path,"must be string","TYPE_ERROR"))
            elif not is_rfc3339_date_time(v): errors.append(_err(path,"must be an RFC 3339 date-time with seconds and an offset, e.g. 2026-01-15T10:30:00Z","INVALID_DATE_TIME"))
        elif kind == "enum":
            allowed = f[2]
            if not isinstance(v,str) or v not in allowed: errors.append(_err(path,"must be one of: "+", ".join(allowed),"INVALID_VALUE"))
        elif kind in ("integer","number"):
            lo = f[2] if len(f) > 2 else None
            hi = f[3] if len(f) > 3 else None
            if not (_is_int(v) if kind == "integer" else _is_num(v)):
                errors.append(_err(path,f"must be {'an integer' if kind == 'integer' else 'a number'}","TYPE_ERROR"))
            elif (lo is not None and v < lo) or (hi is not None and v > hi):
                rng = f">= {lo}" if hi is None else (f"<= {hi}" if lo is None else f"in [{lo},{hi}]")
                errors.append(_err(path,f"must be {rng}","OUT_OF_RANGE"))
        else:  # pragma: no cover -- programming error in a spec table below
            raise ValueError(f"unknown field kind: {kind}")

# spec/schemas/testcase.json + suite.json `provider` (both declare max_tokens minimum: 1).
_CONFIG_PROVIDER_FIELDS = [("model","string"),("api_base","string"),("api_key_env","string"),("temperature","number"),("max_tokens","integer",1),("extra","object")]
# spec/schemas/testcase.json optional properties.
_TESTCASE_FIELDS = [
    ("expected_output","string"),("context","string_array"),("retrieval_context","string_array"),
    ("tools_called","string_array"),("expected_tools","string_array"),("metadata","object"),
    ("tags","string_array"),("provider","object"),("params","object"),
    ("timeout_ms","integer",1),("weight","number",0),
]
# spec/schemas/grader.json optional properties.
_GRADER_FIELDS = [("params","object"),("weight","number",0),("description","string")]
# spec/schemas/grader.json per-type allOf/then optional params.
_GRADER_PARAM_FIELDS = {
    "contains": [("ignore_case","boolean")],
    "regex": [("flags","string")],
    "semantic_similarity": [("model","string"),("provider","string")],
    "llm_judge": [("provider","string"),("temperature","number",0,2),("schema","object")],
    "json_schema": [("strict","boolean")],
    "json_path": [("operator","enum",("eq","ne","gt","lt","gte","lte","contains"))],
    "code": [("timeout_ms","integer",100)],
}
# spec/schemas/suite.json optional properties (graders/test_cases arrays are
# checked inline in validate_suite, since their items are validated there too).
_SUITE_FIELDS = [
    ("$schema","string"),("name","string"),("description","string"),
    ("test_cases_file","string"),("config","object"),("metadata","object"),("tags","string_array"),
]
_SUITE_CONFIG_FIELDS = [("provider","object"),("defaults","object"),("parallel","integer",1),("retry","object")]
_SUITE_DEFAULTS_FIELDS = [("timeout_ms","integer",1),("weight","number",0)]
_SUITE_RETRY_FIELDS = [("max_attempts","integer",1),("backoff_ms","integer",100)]
# spec/schemas/resultset.json optional top-level properties (group/isolation are
# checked separately below).
_RESULTSET_FIELDS = [
    ("$schema","string"),("suite_version","string"),("completed_at","date_time"),
    ("provider","object"),("runner","object"),("summary","object"),("metadata","object"),
]
_RESULTSET_PROVIDER_FIELDS = [("model","string"),("api_base","string"),("temperature","number"),("max_tokens","integer"),("extra","object")]
_RUNNER_FIELDS = [("name","string"),("version","string")]
_SUMMARY_FIELDS = [
    ("total","integer",0),("passed","integer",0),("failed","integer",0),("skipped","integer",0),
    ("pass_rate","number",0,1),("avg_score","number",0,1),("duration_ms","integer",0),("by_grader","object"),
]
_BY_GRADER_ENTRY_FIELDS = [("passed","integer"),("failed","integer"),("avg_score","number")]
_RESULT_FIELDS = [
    ("actual_output","string"),("duration_ms","integer",0),("completed_at","date_time"),
    ("error","object"),("metadata","object"),
]
_RESULT_ERROR_FIELDS = [
    ("type","enum",("timeout","provider_error","runner_error")),("message","string"),
    ("code","string_or_integer"),("retryable","boolean"),
]
_GRADER_RESULT_FIELDS = [("reason","string"),("metadata","object")]

def validate_test_case(tc):
    errors=[]
    if not isinstance(tc,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(tc.get("id"),str) or not tc["id"]: errors.append(_err("$.id","id required","REQUIRED"))
    inp=tc.get("input")
    if not isinstance(inp,str) and not (isinstance(inp,list) and all(isinstance(x,str) for x in inp)): errors.append(_err("$.input","input required","REQUIRED"))
    elif isinstance(inp,str) and not inp: errors.append(_err("$.input","empty","MIN_LENGTH"))
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
    _check_optional(tc,"$",_TESTCASE_FIELDS,errors)
    if isinstance(tc.get("provider"),dict): _check_optional(tc["provider"],"$.provider",_CONFIG_PROVIDER_FIELDS,errors)
    return ValidationResult(not errors,errors)

def validate_grader(g):
    errors=[]
    if not isinstance(g,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(g.get("id"),str) or not g["id"]: errors.append(_err("$.id","id required","REQUIRED"))
    _check_optional(g,"$",_GRADER_FIELDS,errors)
    gt=g.get("type")
    if not isinstance(gt,str) or not gt: errors.append(_err("$.type","type required","REQUIRED"))
    else:
        # A non-object params already produced a TYPE_ERROR above; validate the
        # per-type requirements against {} so they're still reported (and so a
        # list/string params can't crash the .get() calls in _vp).
        p=g.get("params") if isinstance(g.get("params"),dict) else {}
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
        if not _is_num(th) or th<0 or th>1: e.append(_err("threshold","0-1","OUT_OF_RANGE"))
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
    # Optional, typed per-type params (grader.json allOf/then blocks). Paths here are
    # relative to params (validate_grader prefixes "$.params.").
    opt=[]
    _check_optional(p,"$",_GRADER_PARAM_FIELDS.get(t,[]),opt)
    for x in opt: e.append(_err(x["path"][2:],x["message"],x["code"]))
    return e

def validate_suite(s):
    errors=[]
    if not isinstance(s,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(s.get("version"),str) or not SEMVER_RE.match(s.get("version","")): errors.append(_err("$.version","semver","INVALID_VERSION"))
    if not isinstance(s.get("id"),str) or not s["id"]: errors.append(_err("$.id","required","REQUIRED"))
    tcs=s.get("test_cases")
    if not isinstance(tcs,list) and not isinstance(s.get("test_cases_file"),str): errors.append(_err("$.test_cases","required","REQUIRED"))
    elif "test_cases" in s and not isinstance(tcs,list): errors.append(_err("$.test_cases","must be array","TYPE_ERROR"))
    _check_optional(s,"$",_SUITE_FIELDS,errors)
    grs=s.get("graders",[])
    if not isinstance(grs,list): errors.append(_err("$.graders","must be array","TYPE_ERROR"))
    cfg=s.get("config")
    if isinstance(cfg,dict):
        _check_optional(cfg,"$.config",_SUITE_CONFIG_FIELDS,errors)
        if isinstance(cfg.get("provider"),dict): _check_optional(cfg["provider"],"$.config.provider",_CONFIG_PROVIDER_FIELDS,errors)
        if isinstance(cfg.get("defaults"),dict): _check_optional(cfg["defaults"],"$.config.defaults",_SUITE_DEFAULTS_FIELDS,errors)
        if isinstance(cfg.get("retry"),dict): _check_optional(cfg["retry"],"$.config.retry",_SUITE_RETRY_FIELDS,errors)
    # Shared graders are validated whether test cases are inline or in
    # test_cases_file -- previously an invalid grader in a test_cases_file suite
    # was never looked at.
    gids=set()
    if isinstance(grs,list):
        for i,g in enumerate(grs):
            gv=validate_grader(g)
            if not gv.valid:
                for e in gv.errors: errors.append(_err(f"$.graders[{i}].{e['path']}",e["message"],e["code"]))
            gid=g.get("id") if isinstance(g,dict) else None
            if isinstance(gid,str):
                if gid in gids: errors.append(_err(f"$.graders[{i}].id",f"dup:{gid}","DUPLICATE_ID"))
                gids.add(gid)
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
        if isinstance(grs,list):
            for i,tc in enumerate(tcs):
                if isinstance(tc,dict) and isinstance(tc.get("graders"),list):
                    for j,gr in enumerate(tc["graders"]):
                        if isinstance(gr,str) and gr not in gids: errors.append(_err(f"$.test_cases[{i}].graders[{j}]",f"not found:{gr}","DANGLING_REFERENCE"))
    return ValidationResult(not errors,errors)

# ---------------------------------------------------------------------------
# PROPOSED, Discussion #49 (alternative B: declared aggregation), NOT on main.
# A Result whose graders are partly null-scored (some "not verified", some scored)
# reads differently under different aggregation policies, and a ResultSet
# validator never sees the suite. This alternative makes the policy part of the
# document instead of adding a Result.verdict field: the declaration may sit on
# the Result (metadata.openeval.aggregation, dotted or nested) or on the
# ResultSet's own metadata as the run default.
# ---------------------------------------------------------------------------
AGGREGATION_STRATEGIES = ("all", "any", "majority", "weighted", "strict", "producer")
AGGREGATION_STATUSES = ("unscored", "partial")

def _openeval_key(meta, key):
    """Return (present, value, path_suffix) for metadata['openeval.<key>'] or
    metadata['openeval'][<key>]. Both spellings appear in the spec and the SDKs."""
    if not isinstance(meta, dict): return (False, None, None)
    if "openeval."+key in meta: return (True, meta["openeval."+key], "['openeval."+key+"']")
    oe = meta.get("openeval")
    if isinstance(oe, dict) and key in oe: return (True, oe[key], ".openeval."+key)
    return (False, None, None)

def _check_aggregation(agg, path, errors):
    """Shape of a declared openeval.aggregation object (PROPOSED)."""
    if not isinstance(agg, dict):
        errors.append(_err(path, "must be object", "TYPE_ERROR")); return
    st = agg.get("strategy")
    if st not in AGGREGATION_STRATEGIES:
        errors.append(_err(path+".strategy", "must be one of: "+", ".join(AGGREGATION_STRATEGIES), "INVALID_VALUE"))
    if "threshold" in agg:
        th = agg["threshold"]
        if not _is_num(th) or th < 0 or th > 1: errors.append(_err(path+".threshold", "must be in [0,1]", "OUT_OF_RANGE"))
    elif st == "weighted":
        errors.append(_err(path+".threshold", "required for strategy weighted", "REQUIRED"))

def _score_census(grs):
    """(null_scored, number_scored) over well-formed GraderResults."""
    nulls = sum(1 for g in grs if isinstance(g, dict) and "score" in g and g["score"] is None)
    nums = sum(1 for g in grs if isinstance(g, dict) and _is_num(g.get("score")))
    return nulls, nums

def validate_result_set(r):
    errors=[]
    if not isinstance(r,dict): return ValidationResult(False,[_err("$","Must be object","TYPE_ERROR")])
    if not isinstance(r.get("version"),str) or not SEMVER_RE.match(r.get("version","")): errors.append(_err("$.version","semver","INVALID_VERSION"))
    if not isinstance(r.get("suite_id"),str) or not r["suite_id"]: errors.append(_err("$.suite_id","required","REQUIRED"))
    run_id=r.get("run_id")
    if not isinstance(run_id,str) or not run_id: errors.append(_err("$.run_id","required","REQUIRED"))
    started_at=r.get("started_at")
    if not isinstance(started_at,str): errors.append(_err("$.started_at","required","REQUIRED"))
    elif not is_rfc3339_date_time(started_at): errors.append(_err("$.started_at","must be an RFC 3339 date-time with seconds and an offset, e.g. 2026-01-15T10:30:00Z","INVALID_DATE_TIME"))
    _check_optional(r,"$",_RESULTSET_FIELDS,errors)
    if isinstance(r.get("provider"),dict): _check_optional(r["provider"],"$.provider",_RESULTSET_PROVIDER_FIELDS,errors)
    if isinstance(r.get("runner"),dict): _check_optional(r["runner"],"$.runner",_RUNNER_FIELDS,errors)
    summary=r.get("summary")
    if isinstance(summary,dict):
        _check_optional(summary,"$.summary",_SUMMARY_FIELDS,errors)
        if isinstance(summary.get("by_grader"),dict):
            for gk,gv in summary["by_grader"].items():
                if not isinstance(gv,dict): errors.append(_err(f"$.summary.by_grader.{gk}","must be object","TYPE_ERROR"))
                else: _check_optional(gv,f"$.summary.by_grader.{gk}",_BY_GRADER_ENTRY_FIELDS,errors)
    # PROPOSED (Discussion #49, alt B): a run-level aggregation declaration on the
    # ResultSet's own metadata is validated here and used as the default below.
    rs_decl, rs_agg, rs_suffix = _openeval_key(r.get("metadata"), "aggregation")
    if rs_decl: _check_aggregation(rs_agg, "$.metadata"+rs_suffix, errors)
    # isolation / group / group.* / attempt below: like every other optional field,
    # an explicit null is a type error (resultset.json declares none of them nullable).
    if "isolation" in r and not isinstance(r["isolation"],str): errors.append(_err("$.isolation","must be string","TYPE_ERROR"))
    # Discussion #45 / PR #54: optional group membership joining sibling
    # ResultSets (a sweep, a mutation-testing run, a multi-model comparison).
    # Absent by default -- a no-op for every ResultSet produced before this.
    if "group" in r:
        group=r["group"]
        if not isinstance(group,dict): errors.append(_err("$.group","must be object","TYPE_ERROR"))
        else:
            gid=group.get("group_id")
            if not isinstance(gid,str) or not gid: errors.append(_err("$.group.group_id","required","REQUIRED"))
            if "parent_group_id" in group:
                pgid=group["parent_group_id"]
                if not isinstance(pgid,str) or not pgid: errors.append(_err("$.group.parent_group_id","must be a non-empty string","REQUIRED"))
                elif isinstance(gid,str) and pgid==gid: errors.append(_err("$.group.parent_group_id","a group cannot be its own parent","SELF_PARENT"))
            if "role" in group and not isinstance(group["role"],str): errors.append(_err("$.group.role","must be string","TYPE_ERROR"))
            if "label" in group and not isinstance(group["label"],str): errors.append(_err("$.group.label","must be string","TYPE_ERROR"))
            if "sequence" in group:
                seq=group["sequence"]
                if not _is_int(seq) or seq<0: errors.append(_err("$.group.sequence","must be an integer >= 0","OUT_OF_RANGE"))
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
            if not isinstance(x.get("test_case_id"),str) or not x["test_case_id"]: errors.append(_err(f"$.results[{i}].test_case_id","required","REQUIRED"))
            if not isinstance(x.get("passed"),bool): errors.append(_err(f"$.results[{i}].passed","required","REQUIRED"))
            _check_optional(x,f"$.results[{i}]",_RESULT_FIELDS,errors)
            if isinstance(x.get("error"),dict): _check_optional(x["error"],f"$.results[{i}].error",_RESULT_ERROR_FIELDS,errors)
            if "attempt" in x:
                attempt=x["attempt"]
                if not _is_int(attempt) or attempt<1:
                    errors.append(_err(f"$.results[{i}].attempt","must be an integer >= 1","OUT_OF_RANGE"))
                elif isinstance(x.get("test_case_id"),str) and isinstance(run_id,str):
                    # Only build the uniqueness key once test_case_id/run_id are
                    # confirmed strings: an unhashable type (list/dict) here would
                    # raise TypeError on the `in`/`add` below, and either field
                    # being malformed already produces its own structured error
                    # above ($.run_id / $.results[i].test_case_id), so skipping
                    # the dedup check for those entries -- rather than crashing
                    # -- is safe.
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
                    if not isinstance(gr.get("grader_id"),str) or not gr["grader_id"]: errors.append(_err(f"$.results[{i}].grader_results[{j}].grader_id","required","REQUIRED"))
                    if not isinstance(gr.get("type"),str): errors.append(_err(f"$.results[{i}].grader_results[{j}].type","required","REQUIRED"))
                    # score is REQUIRED (resultset.json) -- an absent key is not the
                    # same as an explicit null ("not verified", Rule 6).
                    if "score" not in gr: errors.append(_err(f"$.results[{i}].grader_results[{j}].score","required","REQUIRED"))
                    else:
                        sc=gr["score"]
                        if sc is not None and not _is_num(sc): errors.append(_err(f"$.results[{i}].grader_results[{j}].score","number|null","TYPE_ERROR"))
                        elif sc is not None and (sc<0 or sc>1): errors.append(_err(f"$.results[{i}].grader_results[{j}].score","must be in [0,1] or null","OUT_OF_RANGE"))
                    if not isinstance(gr.get("passed"),bool): errors.append(_err(f"$.results[{i}].grader_results[{j}].passed","required","REQUIRED"))
                    # SPEC.md Validation Rules -> 6. Result Consistency: "Skipped or
                    # not-yet-executed graders (e.g. `human` review pending, an
                    # `unsupported_grader_type`, a runner error before scoring) MUST be
                    # represented with `score: null` and `passed: false`. `score: null`
                    # means "not verified" [...]"
                    elif "score" in gr and gr["score"] is None and gr["passed"] is True:
                        errors.append(_err(f"$.results[{i}].grader_results[{j}].passed","a GraderResult with score null (not verified) MUST have passed false (SPEC Rule 6)","NULL_SCORE_PASSED"))
                    _check_optional(gr,f"$.results[{i}].grader_results[{j}]",_GRADER_RESULT_FIELDS,errors)
                # SPEC.md Extension Mechanism -> Aggregation Extension: "A test case
                # whose graders are *all* null-scored has no basis for a pass/fail
                # verdict; runners MUST report such a case's `passed` as `false` [...]"
                if grs and x.get("passed") is True and all(isinstance(gr,dict) and "score" in gr and gr["score"] is None for gr in grs):
                    errors.append(_err(f"$.results[{i}].passed","a Result whose grader_results are all null-scored MUST have passed false (SPEC Aggregation Extension / Rule 6)","UNSCORED_RESULT_PASSED"))
                # PROPOSED (Discussion #49, alt B: declared aggregation), NOT on main.
                nulls, nums = _score_census(grs)
                mixed = nulls > 0 and nums > 0
                res_decl, res_agg, res_suffix = _openeval_key(x.get("metadata"), "aggregation")
                if res_decl: _check_aggregation(res_agg, f"$.results[{i}].metadata"+res_suffix, errors)
                if mixed and not res_decl and not rs_decl:
                    errors.append(_err(f"$.results[{i}].metadata",
                        "a Result with some null-scored and some scored graders MUST declare how passed was derived: "
                        "openeval.aggregation on the Result or on the ResultSet (PROPOSED, Discussion #49 alt B)",
                        "PARTIAL_RESULT_UNDECLARED"))
                st_present, status, st_suffix = _openeval_key(x.get("metadata"), "aggregation_status")
                if st_present:
                    spath = f"$.results[{i}].metadata"+st_suffix
                    if status not in AGGREGATION_STATUSES:
                        errors.append(_err(spath, "must be one of: "+", ".join(AGGREGATION_STATUSES), "INVALID_VALUE"))
                    elif status == "unscored" and nums > 0:
                        errors.append(_err(spath, "unscored means every grader has score null, but this Result has a scored grader", "AGGREGATION_STATUS_MISMATCH"))
                    elif status == "partial" and not mixed:
                        errors.append(_err(spath, "partial means some but not all graders have score null", "AGGREGATION_STATUS_MISMATCH"))
    return ValidationResult(not errors,errors)

def validate_document(d,t):
    if t=="testcase": return validate_test_case(d)
    if t=="grader": return validate_grader(d)
    if t=="suite": return validate_suite(d)
    if t=="resultset": return validate_result_set(d)
    raise ValueError(f"Unknown type: {t}")
