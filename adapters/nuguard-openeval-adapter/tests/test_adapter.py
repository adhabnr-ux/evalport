from types import SimpleNamespace

import pytest
from openeval.validate import validate_result_set

from nuguard_openeval_adapter import finding_to_result, from_openeval, to_openeval


# ---------------------------------------------------------------------------
# Fixtures mirroring nuguard's real shapes (nuguard/models/finding.py,
# nuguard/models/validate.py). ValidateRunResult.findings is `list[dict]`
# (serialized Finding dicts) per the correction on NuGuardAI/nuguard#355 --
# these fixtures use plain dicts for findings for that reason, matching what
# `[f.model_dump() for f in findings]` in nuguard/validate/runner.py actually
# produces.
# ---------------------------------------------------------------------------


def make_finding(**overrides):
    finding = {
        "finding_id": "f_1",
        "title": "Boundary failure: refund_policy",
        "severity": "high",
        "description": "Boundary assertion 'refund_policy' failed.",
        "affected_component": "refund_policy",
        "remediation": "Review the agent's system prompt.",
        "references": [],
        "container_image": None,
        "container_image_locations": [],
        "goal_type": "BOUNDARY_FAILURE",
        "scenario_type": None,
        "sbom_path": [],
        "sbom_path_descriptions": [],
        "policy_clauses_violated": [],
        "chain_id": None,
        "owasp_asi_ref": None,
        "owasp_llm_ref": "LLM01",
        "mitre_atlas_technique": None,
        "evidence": "response was not a refusal",
        "log_correlation_status": None,
        "reasoning": "",
        "evidence_quote": "",
        "success_indicator": None,
        "scores": {},
        "attack_steps": [],
        "golden_ids": [],
        "golden_name": None,
        "golden_data_excerpt": None,
        "verified": None,
        "ngrs_score": 62,
        "ngrs_vector": "DC:4/VOL:2/SC:1",
        "authorization_decision": "allow",
        "guardrail_control": "",
    }
    finding.update(overrides)
    return finding


def make_capability_map(built_at="2026-08-24T10:00:00+00:00", entries=None):
    return {
        "run_id": "run_abc",
        "built_at": built_at,
        "entries": entries
        if entries is not None
        else [
            {"tool_name": "search_orders", "node_type": "tool", "exercised": True, "policy_compliant": True},
            {"tool_name": "issue_refund", "node_type": "tool", "exercised": False, "policy_compliant": True},
        ],
    }


def make_run_result(findings=None, scan_outcome="findings", **overrides):
    result = {
        "run_id": "run_abc",
        "findings": findings if findings is not None else [make_finding()],
        "capability_map": make_capability_map(),
        "policy_records": [{"turn": 1, "prompt": "hi", "response": "hello", "passed": True}],
        "scenarios_executed": 3,
        "scan_outcome": scan_outcome,
        "effective_endpoint": "/chat",
        "target_endpoint_source": "probe",
    }
    result.update(overrides)
    return result


# ---------------------------------------------------------------------------
# finding_to_result
# ---------------------------------------------------------------------------


def test_finding_to_result_maps_severity_to_score_and_fails_by_default():
    result = finding_to_result(make_finding(severity="high"))
    gr = result["grader_results"][0]
    assert gr["score"] == 0.25
    assert gr["passed"] is False
    assert result["passed"] is False
    assert result["test_case_id"] == "f_1"


@pytest.mark.parametrize(
    "severity,expected_score",
    [("critical", 0.0), ("high", 0.25), ("medium", 0.5), ("low", 0.75), ("info", 1.0)],
)
def test_severity_score_mapping(severity, expected_score):
    result = finding_to_result(make_finding(severity=severity))
    assert result["grader_results"][0]["score"] == expected_score


def test_verified_false_is_the_only_way_a_finding_passes():
    unverified = finding_to_result(make_finding(verified=None))
    reproduced = finding_to_result(make_finding(verified=True))
    disproven = finding_to_result(make_finding(verified=False))
    assert unverified["passed"] is False
    assert reproduced["passed"] is False
    assert disproven["passed"] is True
    assert disproven["grader_results"][0]["passed"] is True


def test_chain_id_preferred_over_finding_id_for_test_case_id():
    result = finding_to_result(make_finding(finding_id="f_1", chain_id="chain_9"))
    assert result["test_case_id"] == "chain_9"


def test_grader_id_derived_from_goal_type():
    result = finding_to_result(make_finding(goal_type="POLICY_VIOLATION"))
    assert result["grader_results"][0]["grader_id"] == "gr_policy_violation"


def test_grader_id_falls_back_when_goal_type_missing():
    result = finding_to_result(make_finding(goal_type=None))
    assert result["grader_results"][0]["grader_id"] == "gr_nuguard_finding"


def test_evidence_quote_preferred_as_reason():
    result = finding_to_result(make_finding(evidence_quote="the exact leaked substring", evidence="broader evidence"))
    assert result["grader_results"][0]["reason"] == "the exact leaked substring"


def test_tags_collect_owasp_and_policy_refs():
    result = finding_to_result(
        make_finding(
            owasp_llm_ref="LLM01",
            mitre_atlas_technique="AML.T0051",
            policy_clauses_violated=["no-pii-disclosure"],
            goal_type="POLICY_VIOLATION",
        )
    )
    tags = result["grader_results"][0]["metadata"]["tags"]
    assert "LLM01" in tags
    assert "AML.T0051" in tags
    assert "policy:no-pii-disclosure" in tags
    assert "goal:POLICY_VIOLATION" in tags


def test_finding_to_result_accepts_attribute_style_object():
    # Some callers may hand the adapter a live Finding pydantic instance
    # rather than a dict -- the accessor must work either way.
    finding_obj = SimpleNamespace(**make_finding(finding_id="f_obj", severity="critical"))
    result = finding_to_result(finding_obj)
    assert result["test_case_id"] == "f_obj"
    assert result["grader_results"][0]["score"] == 0.0


def test_ngrs_fallback_when_severity_unparseable():
    result = finding_to_result(make_finding(severity="unknown-severity", ngrs_score=80))
    assert result["grader_results"][0]["score"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


def test_to_openeval_validates_against_evalport_spec():
    run_result = make_run_result()
    result_set = to_openeval(run_result)
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_to_openeval_one_result_per_finding():
    run_result = make_run_result(findings=[make_finding(finding_id="f_1"), make_finding(finding_id="f_2")])
    result_set = to_openeval(run_result)
    assert len(result_set["results"]) == 2
    assert {r["test_case_id"] for r in result_set["results"]} == {"f_1", "f_2"}


def test_to_openeval_empty_findings_still_produces_valid_resultset():
    run_result = make_run_result(findings=[], scan_outcome="no_findings")
    result_set = to_openeval(run_result)
    assert len(result_set["results"]) == 1
    assert result_set["results"][0]["passed"] is True
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
    assert result_set["summary"]["pass_rate"] == 1.0


def test_to_openeval_carries_capability_map_and_policy_records_as_metadata():
    run_result = make_run_result()
    result_set = to_openeval(run_result)
    nuguard_meta = result_set["metadata"]["nuguard"]
    assert nuguard_meta["scan_outcome"] == "findings"
    assert nuguard_meta["capability_map"]["tools_total"] == 2
    assert nuguard_meta["capability_map"]["tools_exercised"] == 1
    assert nuguard_meta["policy_records_count"] == 1


def test_to_openeval_uses_capability_map_built_at_as_started_at():
    run_result = make_run_result()
    result_set = to_openeval(run_result)
    assert result_set["started_at"] == "2026-08-24T10:00:00+00:00"


def test_to_openeval_suite_id_default_and_override():
    run_result = make_run_result()
    assert to_openeval(run_result)["suite_id"] == "nuguard_validate_run_abc"
    assert to_openeval(run_result, suite_id="custom_suite")["suite_id"] == "custom_suite"


def test_to_openeval_summary_by_grader():
    run_result = make_run_result(
        findings=[
            make_finding(finding_id="f_1", goal_type="POLICY_VIOLATION", severity="critical"),
            make_finding(finding_id="f_2", goal_type="POLICY_VIOLATION", severity="low"),
        ]
    )
    result_set = to_openeval(run_result)
    by_grader = result_set["summary"]["by_grader"]["gr_policy_violation"]
    assert by_grader["failed"] == 2
    assert by_grader["avg_score"] == pytest.approx((0.0 + 0.75) / 2)


def test_to_openeval_accepts_attribute_style_run_result():
    run_result_dict = make_run_result()
    run_result_obj = SimpleNamespace(
        **{**run_result_dict, "capability_map": SimpleNamespace(**run_result_dict["capability_map"])}
    )
    result_set = to_openeval(run_result_obj)
    assert validate_result_set(result_set).valid


def test_to_openeval_scan_outcome_matches_real_validate_runner_vocabulary():
    # nuguard/validate/runner.py only ever produces these four values for a
    # validate run's scan_outcome -- guard against silently accepting typos.
    for outcome in ("no_findings", "findings", "high_findings", "critical_findings"):
        run_result = make_run_result(scan_outcome=outcome, findings=[] if outcome == "no_findings" else None)
        result_set = to_openeval(run_result)
        assert result_set["metadata"]["nuguard"]["scan_outcome"] == outcome
        assert validate_result_set(result_set).valid


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


def test_from_openeval_reconstructs_findings_from_failed_results():
    run_result = make_run_result(
        findings=[make_finding(finding_id="f_1", severity="critical", evidence_quote="leaked SSN 123-45-6789")]
    )
    result_set = to_openeval(run_result)
    findings = from_openeval(result_set)
    assert len(findings) == 1
    assert findings[0]["finding_id"] == "f_1"
    assert findings[0]["severity"] == "critical"
    assert findings[0]["description"] == "leaked SSN 123-45-6789"


def test_from_openeval_skips_passing_results():
    run_result = make_run_result(findings=[make_finding(finding_id="f_1", verified=False)])
    result_set = to_openeval(run_result)
    assert from_openeval(result_set) == []


def test_from_openeval_skips_synthetic_no_findings_result():
    run_result = make_run_result(findings=[], scan_outcome="no_findings")
    result_set = to_openeval(run_result)
    assert from_openeval(result_set) == []


def test_from_openeval_round_trip_preserves_severity_band():
    run_result = make_run_result(
        findings=[
            make_finding(finding_id="f_1", severity="critical"),
            make_finding(finding_id="f_2", severity="medium"),
        ]
    )
    result_set = to_openeval(run_result)
    findings = from_openeval(result_set)
    severities = {f["finding_id"]: f["severity"] for f in findings}
    assert severities["f_1"] == "critical"
    assert severities["f_2"] == "medium"
