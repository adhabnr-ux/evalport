import json
import uuid

import pytest

from agenta.sdk.models.testsets import TestsetRevision, TestsetRevisionData, Testcase
from agenta.sdk.models.workflows import (
    WorkflowInvokeRequest,
    WorkflowRequestData,
    WorkflowBatchResponse,
    WorkflowServiceResponseData,
    WorkflowServiceStatus,
)

from openeval.validate import validate_suite, validate_result_set

from agenta_openeval_adapter import (
    to_openeval,
    from_openeval,
    agenta_testset_to_suite,
    invocations_to_resultset,
)


TC1_ID = "11111111-1111-1111-1111-111111111111"
TC2_ID = "22222222-2222-2222-2222-222222222222"


def _tc(id_, **data):
    return Testcase(id=uuid.UUID(id_), data=data)


def make_testset_revision():
    tc1 = _tc(TC1_ID, input="What is 2+2?", expected_output="4", difficulty="easy")
    tc2 = _tc(TC2_ID, input="Capital of France?", expected_output="Paris")
    return TestsetRevision(
        id=uuid.uuid4(),
        slug="qa-basic",
        name="QA basic",
        data=TestsetRevisionData(testcases=[tc1, tc2]),
    )


# ---------------------------------------------------------------------------
# testset -> suite
# ---------------------------------------------------------------------------


def test_testset_to_suite_basic():
    suite = agenta_testset_to_suite(make_testset_revision())

    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["id"] == TC1_ID
    assert tc1["input"] == "What is 2+2?"
    assert tc1["expected_output"] == "4"
    assert tc1["graders"] == ["gr_output_match"]
    assert tc1["metadata"]["agenta_testcase"] == {"difficulty": "easy"}

    tc2 = suite["test_cases"][1]
    assert "metadata" not in tc2

    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"gr_output_match"}
    assert suite["graders"][0]["type"] == "llm_judge"


def test_testset_to_suite_validates_against_evalport_spec():
    suite = agenta_testset_to_suite(make_testset_revision())
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_testset_to_suite_exact_match_grader_option():
    suite = agenta_testset_to_suite(make_testset_revision(), grader_type="exact_match")
    assert suite["graders"][0]["type"] == "exact_match"


def test_testset_to_suite_from_bare_revision_data():
    # A caller may pass just the TestsetRevisionData (no enclosing TestsetRevision) --
    # e.g. after fetching `testset_revision.data` directly.
    data = TestsetRevisionData(testcases=[_tc(TC1_ID, input="hi", expected_output="hello")])
    suite = agenta_testset_to_suite(data, suite_id="s1")
    assert suite["id"] == "s1"
    assert suite["test_cases"][0]["expected_output"] == "hello"


def test_testset_to_suite_custom_column_names():
    tc = _tc(TC1_ID, question="2+2?", answer="4")
    data = TestsetRevisionData(testcases=[tc])
    suite = agenta_testset_to_suite(data, suite_id="s1", input_key="question", expected_output_key="answer")
    tc_out = suite["test_cases"][0]
    assert tc_out["input"] == "2+2?"
    assert tc_out["expected_output"] == "4"


def test_testset_to_suite_input_fallback_when_no_recognized_column():
    tc = _tc(TC1_ID, foo="bar", baz=1)
    data = TestsetRevisionData(testcases=[tc])
    suite = agenta_testset_to_suite(data, suite_id="s2")
    tc_out = suite["test_cases"][0]
    assert tc_out["input"]
    assert json.loads(tc_out["input"]) == {"foo": "bar", "baz": 1}


def test_testset_to_suite_empty():
    revision = TestsetRevision(id=uuid.uuid4(), slug="empty", data=TestsetRevisionData(testcases=[]))
    suite = agenta_testset_to_suite(revision)
    assert suite["test_cases"] == []
    assert suite["graders"] == []
    assert suite["version"]


def test_testset_to_suite_missing_data():
    # A TestsetRevision with no hydrated `.data` at all (e.g. only testcase_ids
    # were fetched) shouldn't crash -- it should just produce no test cases.
    revision = TestsetRevision(id=uuid.uuid4(), slug="ids-only")
    suite = agenta_testset_to_suite(revision)
    assert suite["test_cases"] == []


def test_to_openeval_dispatches_to_testset():
    suite = to_openeval(make_testset_revision())
    assert suite["id"].startswith("agenta_testset_")
    assert len(suite["test_cases"]) == 2


# ---------------------------------------------------------------------------
# evaluator invocations -> result set
# ---------------------------------------------------------------------------


def make_invocation(tc_id, score, passed, actual_output="4", grader_id="gr_output_match"):
    request = WorkflowInvokeRequest(
        data=WorkflowRequestData(testcase={"id": tc_id}, outputs=actual_output)
    )
    response = WorkflowBatchResponse(
        status=WorkflowServiceStatus(code=200, message="Success"),
        data=WorkflowServiceResponseData(outputs={"score": score, "passed": passed, "reason": "ok"}),
    )
    return {"test_case_id": tc_id, "request": request, "response": response, "grader_id": grader_id}


def test_invocations_to_resultset_basic():
    invocations = [
        make_invocation(TC1_ID, 1.0, True, actual_output="4"),
        make_invocation(TC2_ID, 0.0, False, actual_output="London"),
    ]
    resultset = invocations_to_resultset(invocations, suite_id="agenta_testset_x", run_id="run1")

    assert resultset["suite_id"] == "agenta_testset_x"
    assert resultset["run_id"] == "run1"
    assert len(resultset["results"]) == 2

    r1 = next(r for r in resultset["results"] if r["test_case_id"] == TC1_ID)
    assert r1["passed"] is True
    assert r1["actual_output"] == "4"
    assert r1["grader_results"][0]["score"] == 1.0
    assert r1["grader_results"][0]["grader_id"] == "gr_output_match"

    r2 = next(r for r in resultset["results"] if r["test_case_id"] == TC2_ID)
    assert r2["passed"] is False
    assert r2["actual_output"] == "London"

    assert resultset["summary"]["total"] == 2
    assert resultset["summary"]["passed"] == 1
    assert resultset["summary"]["pass_rate"] == 0.5


def test_invocations_to_resultset_validates_against_evalport_spec():
    invocations = [make_invocation(TC1_ID, 1.0, True)]
    resultset = invocations_to_resultset(invocations, suite_id="s1", run_id="run1")
    result = validate_result_set(resultset)
    assert result.valid, result.errors


def test_invocations_groups_multiple_graders_per_test_case():
    request = WorkflowInvokeRequest(data=WorkflowRequestData(testcase={"id": TC1_ID}, outputs="4"))
    resp_ok = WorkflowBatchResponse(
        status=WorkflowServiceStatus(), data=WorkflowServiceResponseData(outputs={"score": 1.0, "passed": True})
    )
    resp_bad = WorkflowBatchResponse(
        status=WorkflowServiceStatus(), data=WorkflowServiceResponseData(outputs={"score": 0.2, "passed": False})
    )
    invocations = [
        {"test_case_id": TC1_ID, "request": request, "response": resp_ok, "grader_id": "gr_exact"},
        {"test_case_id": TC1_ID, "request": request, "response": resp_bad, "grader_id": "gr_judge"},
    ]
    resultset = invocations_to_resultset(invocations, suite_id="s1", run_id="run2")
    assert len(resultset["results"]) == 1
    r = resultset["results"][0]
    assert {gr["grader_id"] for gr in r["grader_results"]} == {"gr_exact", "gr_judge"}
    assert r["passed"] is False  # not all graders passed


def test_invocations_to_resultset_error_status_propagates():
    request = WorkflowInvokeRequest(data=WorkflowRequestData(testcase={"id": TC1_ID}))
    response = WorkflowBatchResponse(status=WorkflowServiceStatus(code=500, message="evaluator crashed"))
    invocations = [{"test_case_id": TC1_ID, "request": request, "response": response, "grader_id": "gr_x"}]
    resultset = invocations_to_resultset(invocations, suite_id="s1", run_id="run3")
    r = resultset["results"][0]
    assert r["passed"] is False
    assert r["error"]["message"] == "evaluator crashed"


def test_invocations_to_resultset_bool_and_numeric_outputs():
    resp_bool = WorkflowBatchResponse(
        status=WorkflowServiceStatus(), data=WorkflowServiceResponseData(outputs=True)
    )
    resp_num = WorkflowBatchResponse(
        status=WorkflowServiceStatus(), data=WorkflowServiceResponseData(outputs=0.9)
    )
    invocations = [
        {"test_case_id": TC1_ID, "response": resp_bool, "grader_id": "gr_bool"},
        {"test_case_id": TC2_ID, "response": resp_num, "grader_id": "gr_num"},
    ]
    resultset = invocations_to_resultset(invocations, suite_id="s1", run_id="run4")
    r1 = next(r for r in resultset["results"] if r["test_case_id"] == TC1_ID)
    r2 = next(r for r in resultset["results"] if r["test_case_id"] == TC2_ID)
    assert r1["grader_results"][0]["score"] == 1.0
    assert r1["passed"] is True
    assert r2["grader_results"][0]["score"] == 0.9
    assert r2["passed"] is True


def test_invocations_to_resultset_score_is_clamped_to_unit_range():
    response = WorkflowBatchResponse(
        status=WorkflowServiceStatus(), data=WorkflowServiceResponseData(outputs={"score": 5, "passed": True})
    )
    invocations = [{"test_case_id": TC1_ID, "response": response, "grader_id": "gr_x"}]
    resultset = invocations_to_resultset(invocations, suite_id="s1", run_id="run5")
    assert resultset["results"][0]["grader_results"][0]["score"] == 1.0
    # Must still validate -- validate_result_set() rejects scores outside [0, 1].
    assert validate_result_set(resultset).valid


def test_invocations_to_resultset_streaming_response_rejected():
    from agenta.sdk.models.workflows import WorkflowStreamingResponse

    async def _gen():
        yield {}

    streaming = WorkflowStreamingResponse(generator=_gen)
    invocations = [{"test_case_id": TC1_ID, "response": streaming, "grader_id": "gr_x"}]
    with pytest.raises(TypeError):
        invocations_to_resultset(invocations, suite_id="s1", run_id="run6")


def test_invocations_to_resultset_empty_shape():
    resultset = invocations_to_resultset([], suite_id="s1", run_id="run7")
    assert resultset["results"] == []
    assert "summary" not in resultset
    assert resultset["suite_id"] == "s1"


def test_to_openeval_dispatches_to_resultset():
    invocations = [make_invocation(TC1_ID, 1.0, True)]
    resultset = to_openeval(invocations, suite_id="s1", run_id="run8")
    assert resultset["run_id"] == "run8"


# ---------------------------------------------------------------------------
# from_openeval / round trip
# ---------------------------------------------------------------------------


def test_from_openeval_round_trip():
    suite = agenta_testset_to_suite(make_testset_revision())
    rebuilt = from_openeval(suite)

    assert len(rebuilt["testcases"]) == 2
    tc1 = rebuilt["testcases"][0]
    assert tc1["id"] == TC1_ID
    assert tc1["data"] == {"input": "What is 2+2?", "expected_output": "4", "difficulty": "easy"}

    # Feed straight back into the real Agenta pydantic models.
    testcases = [Testcase(**tc) for tc in rebuilt["testcases"]]
    data = TestsetRevisionData(testcases=testcases)
    assert len(data.testcases) == 2
    assert str(data.testcases[0].id) == TC1_ID


def test_from_openeval_generates_deterministic_uuid_for_non_uuid_ids():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc_1", "input": "hi", "expected_output": "hello", "graders": ["g1"]}],
    }
    rebuilt = from_openeval(suite)
    tc = rebuilt["testcases"][0]
    assert tc["id"] is not None
    uuid.UUID(tc["id"])  # does not raise

    rebuilt_again = from_openeval(suite)
    assert rebuilt_again["testcases"][0]["id"] == tc["id"]  # deterministic, not random each call


def test_from_openeval_empty_suite():
    suite = {"version": "1.0.0", "id": "s1", "graders": [], "test_cases": []}
    rebuilt = from_openeval(suite)
    assert rebuilt["testcases"] == []
    assert rebuilt["testcase_ids"] is None
