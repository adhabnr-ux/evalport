"""Tests for qwed-openeval-adapter.

Framework-free tests run in EvalPort's min-mode CI (evalport-sdk only). They
use the JSON fixtures in tests/fixtures/, which were generated with qwed 7.2.1
through its public surface (see tests/fixtures/make_fixtures.py). Tests that
need the real `qwed` package (qwed_sdk) skip via find_spec when it's absent;
raw-JSON-Schema tests skip when jsonschema isn't installed.
"""
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

import qwed_openeval_adapter as A
from openeval.types import OPENEVAL_VERSION
from openeval.validate import validate_result_set, validate_suite

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")
ADAPTER_SRC = os.path.join(HERE, "..", "src", "qwed_openeval_adapter", "__init__.py")
SCHEMA_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "spec", "schemas"))

HAS_QWED = importlib.util.find_spec("qwed_sdk") is not None
HAS_JSONSCHEMA = (
    importlib.util.find_spec("jsonschema") is not None
    and importlib.util.find_spec("referencing") is not None
    and os.path.isdir(SCHEMA_DIR)
)
requires_qwed = pytest.mark.skipif(not HAS_QWED, reason="qwed (qwed_sdk) not installed")
requires_no_qwed = pytest.mark.skipif(HAS_QWED, reason="only meaningful without qwed installed")
requires_jsonschema = pytest.mark.skipif(
    not HAS_JSONSCHEMA, reason="jsonschema/referencing not installed or repo schemas not found"
)


def _load(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


VERIFIED_ADMIT = _load("vc_sql_verified_admit.json")
VERIFIED_DENY = _load("vc_code_verified_deny.json")
UNVERIFIABLE = _load("vc_fact_unverifiable.json")
BLOCKED = _load("vc_math_blocked_no_attestation.json")
ALL_DOCS = [VERIFIED_ADMIT, VERIFIED_DENY, UNVERIFIABLE, BLOCKED]
IDS = ["sql-orders", "code-shell", "fact-eiffel", "math-blocked"]


def _rs(docs=ALL_DOCS, ids=IDS, **kw):
    kw.setdefault("run_id", "run-1")
    kw.setdefault("started_at", "2026-09-29T00:00:00Z")
    return A.to_openeval(docs, suite_id="qwed-suite", test_case_ids=ids, **kw)


def _graders(result):
    return {g["grader_id"]: g for g in result["grader_results"]}


def _raw_validators():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    schemas = {}
    for name in ("testcase", "grader", "suite", "resultset"):
        with open(os.path.join(SCHEMA_DIR, f"{name}.json"), encoding="utf-8") as f:
            schemas[name] = json.load(f)
    registry = Registry().with_resources(
        [(s["$id"], Resource.from_contents(s)) for s in schemas.values()]
    )
    return (
        Draft202012Validator(schemas["suite"], registry=registry),
        Draft202012Validator(schemas["resultset"], registry=registry),
    )


# ---------------------------------------------------------------------------
# The mapping, case by case
# ---------------------------------------------------------------------------


def test_verified_admit():
    r = _rs([VERIFIED_ADMIT], ["t"])["results"][0]
    g = _graders(r)
    assert g["qwed.verdict"]["score"] == 1.0 and g["qwed.verdict"]["passed"] is True
    assert g["qwed.admission"]["score"] == 1.0 and g["qwed.admission"]["passed"] is True
    assert r["passed"] is True
    assert r["metadata"]["qwed"]["verdict"] == "VERIFIED"
    assert r["metadata"]["qwed"]["admission"] == "ADMIT"
    assert r["metadata"]["qwed"]["proof_ref"] == VERIFIED_ADMIT["context"]["evidence"]["proof_ref"]


def test_verified_deny_keeps_verdict_passed_but_result_fails():
    """ADR-003's VERIFIED-as-unsafe case must stay representable."""
    r = _rs([VERIFIED_DENY], ["t"])["results"][0]
    g = _graders(r)
    assert g["qwed.verdict"]["score"] == 1.0 and g["qwed.verdict"]["passed"] is True
    assert g["qwed.admission"]["score"] == 0.0 and g["qwed.admission"]["passed"] is False
    assert r["passed"] is False
    assert g["qwed.admission"]["metadata"]["qwed"]["admission"] == "DENY"


@pytest.mark.parametrize("doc,verdict", [(UNVERIFIABLE, "UNVERIFIABLE"), (BLOCKED, "BLOCKED")])
def test_non_verified_is_null_score_not_scored_failure(doc, verdict):
    r = _rs([doc], ["t"])["results"][0]
    g = _graders(r)
    # SPEC Validation Rule 6: not verified -> score null AND passed false.
    assert g["qwed.verdict"]["score"] is None
    assert g["qwed.verdict"]["passed"] is False
    assert g["qwed.verdict"]["metadata"]["qwed"]["verdict"] == verdict
    # QWED fail-closed: UNVERIFIABLE/BLOCKED always DENY, a definitive decision.
    assert g["qwed.admission"]["score"] == 0.0 and g["qwed.admission"]["passed"] is False
    assert r["passed"] is False
    assert r["metadata"]["qwed"]["proof_ref"] is None


def test_scores_are_never_strings():
    for r in _rs()["results"]:
        for g in r["grader_results"]:
            assert g["score"] is None or (isinstance(g["score"], float) and 0.0 <= g["score"] <= 1.0)
            assert isinstance(g["passed"], bool)


def test_result_passed_equals_evalport_default_all_aggregation():
    """Result.passed must agree with SPEC's `all` strategy over non-null-scored graders."""
    for r in _rs()["results"]:
        scored = [g for g in r["grader_results"] if g["score"] is not None]
        assert scored, "admission grader is always scored, so no Result is 'unscored'"
        assert r["passed"] is all(g["passed"] for g in scored)


def test_verdict_and_admission_are_independent_graders():
    for r in _rs()["results"]:
        assert [g["grader_id"] for g in r["grader_results"]] == ["qwed.verdict", "qwed.admission"]
        assert [g["type"] for g in r["grader_results"]] == [
            "qwed_verification_context",
            "qwed_admission_decision",
        ]


def test_reason_explains_passed_semantics():
    g = _graders(_rs([VERIFIED_DENY], ["t"])["results"][0])
    assert "definitively established" in g["qwed.verdict"]["reason"]
    assert "not admissible" in g["qwed.admission"]["reason"]
    g = _graders(_rs([UNVERIFIABLE], ["t"])["results"][0])
    assert "not a scored failure" in g["qwed.verdict"]["reason"]
    assert UNVERIFIABLE["context"]["evidence"]["evidence"]["agent_message"] in g["qwed.verdict"]["reason"]


def test_full_verification_context_embedded_verbatim_and_copied():
    docs = copy.deepcopy(ALL_DOCS)
    rs = _rs(docs)
    for doc, r in zip(ALL_DOCS, rs["results"]):
        assert _graders(r)["qwed.verdict"]["metadata"]["qwed"]["verification_context"] == doc
    rs["results"][0]["grader_results"][0]["metadata"]["qwed"]["verification_context"]["verdict"] = "X"
    assert docs == ALL_DOCS  # output is a deep copy, input untouched


def test_resultset_envelope():
    rs = _rs()
    assert rs["version"] == OPENEVAL_VERSION
    assert rs["suite_id"] == "qwed-suite" and rs["run_id"] == "run-1"
    assert [r["test_case_id"] for r in rs["results"]] == IDS
    assert rs["summary"] == {"total": 4, "passed": 1, "failed": 3, "pass_rate": 0.25}
    q = rs["metadata"]["qwed"]
    assert q["verdict_counts"] == {"VERIFIED": 2, "UNVERIFIABLE": 1, "BLOCKED": 1}
    assert q["admission_counts"] == {"ADMIT": 1, "DENY": 3}
    assert rs["results"][0]["actual_output"] == VERIFIED_ADMIT["object"]["formal_statement"]


def test_default_ids_run_id_and_timestamp():
    rs = A.to_openeval(ALL_DOCS)
    assert [r["test_case_id"] for r in rs["results"]] == ["qwed-1", "qwed-2", "qwed-3", "qwed-4"]
    assert rs["run_id"].startswith("qwed-") and rs["started_at"].endswith("Z")
    assert validate_result_set(rs).valid


def test_json_serializable():
    rs = _rs()
    assert json.loads(json.dumps(rs)) == rs


# ---------------------------------------------------------------------------
# Validation: SDK validators and raw JSON Schemas
# ---------------------------------------------------------------------------


def test_resultset_passes_sdk_validator():
    v = validate_result_set(_rs())
    assert v.valid, v.errors


def test_suite_passes_sdk_validator():
    suite = A.suite_to_openeval(ALL_DOCS, suite_id="qwed-suite", test_case_ids=IDS, name="QWED")
    v = validate_suite(suite)
    assert v.valid, v.errors
    assert suite["test_cases"][0]["input"] == "Total of every order placed by customer 42"
    assert suite["test_cases"][1]["input"] == VERIFIED_DENY["object"]["formal_statement"]
    assert {g["id"] for g in suite["graders"]} == {"qwed.verdict", "qwed.admission"}


@requires_jsonschema
def test_resultset_and_suite_pass_raw_json_schemas():
    suite_v, rs_v = _raw_validators()
    rs = _rs()
    assert not list(rs_v.iter_errors(rs))
    suite = A.suite_to_openeval(ALL_DOCS, suite_id="qwed-suite", test_case_ids=IDS)
    assert not list(suite_v.iter_errors(suite))


def _old_string_score_shape():
    """The earlier (withdrawn) #325 design: QWED strings in GraderResult.score."""
    rs = _rs([VERIFIED_DENY], ["t"])
    g = rs["results"][0]["grader_results"]
    g[0]["score"] = "VERIFIED"
    g[1]["score"] = "DENY"
    return rs


def test_regression_old_string_score_design_fails_sdk_validator():
    """Guard against re-introducing verdict strings in `score` (SPEC Rule 5)."""
    v = validate_result_set(_old_string_score_shape())
    assert not v.valid
    assert any("score" in e["path"] for e in v.errors)


@requires_jsonschema
def test_regression_old_string_score_design_fails_raw_schema():
    _, rs_v = _raw_validators()
    errors = list(rs_v.iter_errors(_old_string_score_shape()))
    assert errors and all("score" in list(e.absolute_path) for e in errors)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_round_trip_documents_unchanged():
    assert A.from_openeval(_rs()) == ALL_DOCS


def test_round_trip_through_json_text():
    text = json.dumps(_rs())
    assert A.from_openeval(json.loads(text)) == ALL_DOCS


def test_round_trip_is_stable():
    rs1 = _rs()
    rs2 = _rs(A.from_openeval(rs1))
    assert rs1 == rs2


def test_from_openeval_rejects_edited_score():
    rs = _rs()
    rs["results"][1]["grader_results"][1]["score"] = 1.0  # DENY edited to look admitted
    with pytest.raises(ValueError, match="qwed.admission"):
        A.from_openeval(rs)


def test_from_openeval_rejects_edited_result_passed():
    rs = _rs()
    rs["results"][1]["passed"] = True
    with pytest.raises(ValueError, match="disagrees"):
        A.from_openeval(rs)


def test_from_openeval_rejects_foreign_resultset():
    foreign = {
        "results": [
            {"test_case_id": "x", "passed": True,
             "grader_results": [{"grader_id": "exact", "type": "exact_match", "score": 1.0, "passed": True}]}
        ]
    }
    with pytest.raises(ValueError, match="qwed.verdict"):
        A.from_openeval(foreign)


# ---------------------------------------------------------------------------
# Input checking
# ---------------------------------------------------------------------------


def _mutated(doc, fn):
    d = copy.deepcopy(doc)
    fn(d)
    return d


@pytest.mark.parametrize(
    "bad,match",
    [
        (_mutated(UNVERIFIABLE, lambda d: d["context"]["decision"].update(admission="ADMIT")), "admission DENY"),
        (_mutated(VERIFIED_ADMIT, lambda d: d["context"]["evidence"].update(proof_ref=None)), "proof_ref"),
        (_mutated(BLOCKED, lambda d: d["context"]["evidence"].update(proof_ref="sha256:" + "a" * 64)), "null"),
        (_mutated(VERIFIED_ADMIT, lambda d: d.update(verdict="REFUTED")), "verdict"),
        (_mutated(VERIFIED_ADMIT, lambda d: d.update(spec_version="2.0")), "spec_version"),
        (_mutated(VERIFIED_ADMIT, lambda d: d["context"]["decision"].update(admission="MAYBE")), "admission"),
        (_mutated(VERIFIED_ADMIT, lambda d: d["object"].update(formal_statement=" ")), "formal_statement"),
    ],
)
def test_structural_invariants_rejected(bad, match):
    with pytest.raises(ValueError, match=match):
        A.to_openeval([bad], validate=False)


def test_legacy_verification_result_shape_rejected():
    class LegacyResult:  # the shape of qwed_sdk.models.VerificationResult
        status = "VERIFIED"
        is_verified = True

    with pytest.raises(TypeError, match="Verification Context"):
        A.to_openeval([LegacyResult()])
    with pytest.raises(ValueError, match="spec_version"):
        A.to_openeval([{"status": "VERIFIED", "is_verified": True}])


def test_test_case_id_checks():
    with pytest.raises(ValueError, match="test_case_ids"):
        A.to_openeval(ALL_DOCS, test_case_ids=["a"])
    with pytest.raises(ValueError, match="unique"):
        A.to_openeval(ALL_DOCS, test_case_ids=["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="at least one"):
        A.to_openeval([])


def test_accepts_objects_with_to_dict():
    class Doc:
        def to_dict(self):
            return copy.deepcopy(VERIFIED_ADMIT)

    assert A.from_openeval(A.to_openeval([Doc()])) == [VERIFIED_ADMIT]


# ---------------------------------------------------------------------------
# Dependency boundary
# ---------------------------------------------------------------------------


def test_adapter_never_imports_qwed_new():
    with open(ADAPTER_SRC, encoding="utf-8") as f:
        src = f.read()
    code_lines = [ln for ln in src.splitlines() if ln.lstrip().startswith(("import ", "from "))]
    assert not [ln for ln in code_lines if "qwed_new" in ln]


def test_importing_adapter_does_not_import_qwed():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, qwed_openeval_adapter; "
         "print(any(m == 'qwed_sdk' or m.startswith(('qwed_sdk.', 'qwed_new')) for m in sys.modules))"],
        capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"


@requires_no_qwed
def test_structural_label_without_qwed_and_validate_true_needs_qwed():
    rs = _rs()
    assert {r["metadata"]["qwed"]["vc_validation"] for r in rs["results"]} == {"structural"}
    with pytest.raises(ImportError, match="qwed"):
        A.to_openeval(ALL_DOCS, validate=True)


# ---------------------------------------------------------------------------
# Against the real qwed package (skipped in EvalPort's min-mode CI)
# ---------------------------------------------------------------------------


@requires_qwed
def test_fixtures_are_real_valid_vc_documents():
    from qwed_sdk import resolve_document_proof_ref, validate_document

    for doc in ALL_DOCS:
        validate_document(doc)
    assert resolve_document_proof_ref(VERIFIED_ADMIT)
    assert resolve_document_proof_ref(VERIFIED_DENY)


@requires_qwed
def test_real_qwed_sdk_documents_all_outcomes():
    from qwed_sdk import (
        Admission, Decision, Evidence, Interpretation, Proof, VerificationContext,
        VerificationContextDocument, resolve_document_proof_ref, validate_document,
    )

    def ctx(admission, **interp):
        return VerificationContext(
            interpretation=Interpretation(**interp),
            proof=Proof(verifier="symbolic", verifier_version="7.2.1"),
            evidence=Evidence(payload={"agent_message": "checked"}),
            decision=Decision(admission=admission),
        )

    docs = [
        VerificationContextDocument.verified(formal_statement="x + 0 = x", context=ctx(Admission.ADMIT, algebra_domain="real")),
        VerificationContextDocument.verified(formal_statement="x / 0 = 1", context=ctx(Admission.DENY, algebra_domain="real")),
        VerificationContextDocument.unverifiable(formal_statement="P = NP", context=ctx(Admission.DENY, theory="ZFC")),
        VerificationContextDocument.blocked(formal_statement="halts(f)", context=ctx(Admission.DENY, theory="ZFC")),
    ]
    rs = A.to_openeval(docs, run_id="r", started_at="2026-09-29T00:00:00Z")
    assert validate_result_set(rs).valid
    assert [r["passed"] for r in rs["results"]] == [True, False, False, False]
    assert [_graders(r)["qwed.verdict"]["score"] for r in rs["results"]] == [1.0, 1.0, None, None]
    assert [_graders(r)["qwed.admission"]["score"] for r in rs["results"]] == [1.0, 0.0, 0.0, 0.0]
    assert {r["metadata"]["qwed"]["vc_validation"] for r in rs["results"]} == {"qwed_sdk.validate_document"}

    back = A.from_openeval(rs, validate=True)
    assert back == [d.to_dict() for d in docs]
    for doc in back:
        validate_document(doc)
    assert resolve_document_proof_ref(back[0]) and resolve_document_proof_ref(back[1])


@requires_qwed
def test_real_qwed_rejects_tampered_proof_ref_commitment():
    tampered = copy.deepcopy(VERIFIED_DENY)
    tampered["object"]["formal_statement"] = "print('hello')"  # proof_ref no longer resolves
    A.to_openeval([tampered], validate=False)  # structurally fine
    with pytest.raises(ValueError, match="qwed_sdk.validate_document"):
        A.to_openeval([tampered])


@requires_qwed
def test_real_legacy_verification_result_rejected():
    from qwed_sdk import VerificationResult

    legacy = VerificationResult(status="VERIFIED", is_verified=True)
    with pytest.raises(TypeError, match="create_verification_context_from_diagnostic"):
        A.to_openeval([legacy])


@requires_qwed
@pytest.mark.skipif(shutil.which("qwed") is None, reason="qwed CLI not on PATH")
def test_real_qwed_cli_from_diagnostic_output():
    diagnostic = {"status": "UNVERIFIABLE", "agent_message": "Could not decide.",
                  "developer_fields": {}, "proof_ref": None}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(diagnostic, f)
        path = f.name
    try:
        out = subprocess.run(
            ["qwed", "context", "from-diagnostic", "--diagnostic-file", path,
             "--query", "SELECT * FROM users", "--verifier", "sql"],
            capture_output=True, text=True, check=True, timeout=120,
        )
    finally:
        os.unlink(path)
    doc = json.loads(out.stdout)
    rs = A.to_openeval([doc], validate=True)
    assert validate_result_set(rs).valid
    g = _graders(rs["results"][0])
    assert g["qwed.verdict"]["score"] is None and g["qwed.admission"]["score"] == 0.0
    assert A.from_openeval(rs) == [doc]
