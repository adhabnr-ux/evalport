from openeval.converters_crewai import from_crewai, crewai_result_to_result_set
from openeval.validate import validate_suite, validate_result_set


CREW_TASKS = {
    "tasks": [
        {
            "id": "task_1",
            "description": "Find the top 3 competitors for our product",
            "expected_output": "A list of 3 competitor names with one-line descriptions",
            "tools": ["search_tool"],
            "agent": "researcher",
        },
        {
            "id": "task_2",
            "description": "Summarize the competitor research into a memo",
            "expected_output": "A 3-paragraph memo",
            "agent": "writer",
        },
        {
            "id": "task_3",
            "description": "Just do a thing with no expectations",
        },
    ]
}


def test_from_crewai_produces_valid_suite():
    suite = from_crewai(CREW_TASKS)
    result = validate_suite(suite)
    assert result.valid, result.errors


def test_from_crewai_maps_tools_and_output():
    suite = from_crewai(CREW_TASKS)
    tcs = {tc["id"]: tc for tc in suite["test_cases"]}

    tc1 = tcs["task_1"]
    assert tc1["input"] == "Find the top 3 competitors for our product"
    assert tc1["expected_tools"] == ["search_tool"]
    assert tc1["expected_output"].startswith("A list")
    assert len(tc1["graders"]) == 2  # output judge + tool check

    tc2 = tcs["task_2"]
    assert "expected_tools" not in tc2
    assert len(tc2["graders"]) == 1  # output judge only

    tc3 = tcs["task_3"]
    assert tc3["graders"][0].startswith("gr_2_default")


def test_crewai_result_to_result_set_is_valid_and_scores_tools():
    suite = from_crewai(CREW_TASKS)
    crew_result = {
        "tasks": [
            {"id": "task_1", "output": "Acme, Globex, Initech", "tools_called": ["search_tool"], "duration_ms": 500},
            {"id": "task_2", "output": "A memo.", "tools_called": [], "duration_ms": 300},
            {"id": "task_3", "output": "Did the thing.", "tools_called": [], "duration_ms": 100},
        ]
    }
    result_set = crewai_result_to_result_set(crew_result, suite, run_id="run_1")

    rs_validation = validate_result_set(result_set)
    assert rs_validation.valid, rs_validation.errors

    results = {r["test_case_id"]: r for r in result_set["results"]}
    tool_grader_result = next(
        gr for gr in results["task_1"]["grader_results"] if gr["grader_id"].endswith("_tools")
    )
    assert tool_grader_result["passed"] is True
    assert tool_grader_result["score"] == 1.0


def test_crewai_result_to_result_set_flags_missing_tool_call():
    suite = from_crewai(CREW_TASKS)
    crew_result = {"tasks": [{"id": "task_1", "output": "Acme only", "tools_called": [], "duration_ms": 500}]}
    result_set = crewai_result_to_result_set(crew_result, suite, run_id="run_2")

    tool_grader_result = next(
        gr for gr in result_set["results"][0]["grader_results"] if gr["grader_id"].endswith("_tools")
    )
    assert tool_grader_result["passed"] is False
    assert tool_grader_result["score"] == 0.0
