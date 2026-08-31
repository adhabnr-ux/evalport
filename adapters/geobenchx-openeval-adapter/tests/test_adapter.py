import pytest

from openeval.validate import validate_suite, validate_result_set, validate_grader

from geobenchx_openeval_adapter import (
    to_openeval,
    results_to_openeval,
    from_openeval,
    GEOBENCHX_CITATION,
    JUDGE_ID,
)

from geobenchx.dataclasses import Task, Solution, Step, TaskSet
from geobenchx.constants import ScoreValues, TaskLabels


def _multi_step_solution():
    return Solution(
        steps=[
            Step(
                function_name="load_geodata",
                arguments={"geodataset": "Railway Network of North America", "output_geodataframe_name": "railways"},
            ),
            Step(
                function_name="calculate_line_lengths",
                arguments={"geodataframe_name": "railways", "output_variable_name": "railway_lengths"},
                comment="units default to km",
            ),
        ]
    )


def _reject_solution():
    return Solution(steps=[Step(function_name="reject_task", arguments={})])


def _solvable_task():
    return Task(
        task_ID="TASK_001",
        task_text="What is the total length of railways in North America?",
        task_labels=[TaskLabels.TASK_SET_01, TaskLabels.SPATIAL_OPERATIONS],
        reference_solution_description="A single load + measure pass is sufficient.",
        reference_solutions=[_multi_step_solution()],
    )


def _unsolvable_task():
    return Task(
        task_ID="TASK_002",
        task_text="What is the meaning of life?",
        task_labels=[TaskLabels.CONTROL],
        reference_solutions=[_reject_solution()],
    )


def _basic_task_set():
    return TaskSet(
        metadata={"model": "gpt-4.1-2025-04-14", "temperature": 0},
        tasks=[_solvable_task(), _unsolvable_task()],
    )


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------

def test_to_openeval_produces_valid_suite():
    suite = to_openeval(_basic_task_set())
    validation = validate_suite(suite)
    assert validation.valid, validation.errors
    assert suite["id"] == "geobenchx"
    assert len(suite["test_cases"]) == 2


def test_to_openeval_maps_solvable_task_fields():
    suite = to_openeval(_basic_task_set())
    tc = suite["test_cases"][0]

    assert tc["id"] == "TASK_001"
    assert tc["input"] == "What is the total length of railways in North America?"
    assert set(tc["tags"]) == {"Task Set 01", "Spatial operations"}
    assert tc["expected_tools"] == ["calculate_line_lengths", "load_geodata"]
    assert tc["graders"] == [JUDGE_ID]

    ref_solutions = tc["metadata"]["reference_solutions"]
    assert len(ref_solutions) == 1
    assert [s["function_name"] for s in ref_solutions[0]] == ["load_geodata", "calculate_line_lengths"]
    assert ref_solutions[0][1]["comment"] == "units default to km"
    assert tc["metadata"]["reference_solution_description"] == "A single load + measure pass is sufficient."
    assert tc["metadata"]["unsolvable"] is False


def test_to_openeval_marks_reject_task_only_as_unsolvable():
    suite = to_openeval(_basic_task_set())
    tc = suite["test_cases"][1]

    assert tc["expected_tools"] == ["reject_task"]
    assert tc["metadata"]["unsolvable"] is True
    assert tc["tags"] == ["Control question"]


def test_to_openeval_flattens_expected_tools_across_multiple_reference_solutions():
    task = Task(
        task_ID="TASK_003",
        task_text="Make a heatmap of population in earthquake zones",
        reference_solutions=[
            Solution(steps=[
                Step(function_name="load_geodata", arguments={}),
                Step(function_name="get_values_from_raster_with_geometries", arguments={}),
                Step(function_name="make_heatmap", arguments={}),
            ]),
            _reject_solution(),
        ],
    )
    suite = to_openeval(TaskSet(tasks=[task]))
    tc = suite["test_cases"][0]

    assert tc["expected_tools"] == [
        "get_values_from_raster_with_geometries",
        "load_geodata",
        "make_heatmap",
        "reject_task",
    ]
    # Multiple reference solutions present -> not the reject-only case.
    assert tc["metadata"]["unsolvable"] is False
    assert len(tc["metadata"]["reference_solutions"]) == 2


def test_to_openeval_carries_taskset_metadata():
    suite = to_openeval(_basic_task_set())
    assert suite["metadata"]["geobenchx_metadata"] == {"model": "gpt-4.1-2025-04-14", "temperature": 0}


def test_to_openeval_includes_citation_matching_citation_cff():
    suite = to_openeval(_basic_task_set())
    citation = suite["metadata"]["citation"]
    assert citation == GEOBENCHX_CITATION
    assert citation["doi"] == "10.1145/3764915.3770721"
    assert "Krechetova" in citation["text"]
    assert "Kochedykov" in citation["text"]
    assert "GeoGenAgent" in citation["text"]


def test_to_openeval_missing_task_id_raises():
    task = Task(task_ID="", task_text="x")
    with pytest.raises(ValueError):
        to_openeval(TaskSet(tasks=[task]))


def test_judge_grader_is_valid_llm_judge_with_taxonomy():
    suite = to_openeval(_basic_task_set())
    grader = suite["graders"][0]

    assert grader["id"] == JUDGE_ID
    assert grader["type"] == "llm_judge"
    gv = validate_grader(grader)
    assert gv.valid, gv.errors

    prompt = grader["params"]["prompt"]
    assert "{input}" in prompt or "{output}" in prompt  # SPEC.md llm_judge token requirement
    assert "Matching score = 0" in prompt
    assert "Matching score = 2" in prompt


def test_to_openeval_accepts_plain_list_not_just_taskset():
    # Duck-typed: works with a bare list of Task objects too, not just TaskSet.
    suite = to_openeval([_solvable_task()])
    validation = validate_suite(suite)
    assert validation.valid, validation.errors
    assert len(suite["test_cases"]) == 1


# ---------------------------------------------------------------------------
# results_to_openeval
# ---------------------------------------------------------------------------

def _scored_task_set():
    t1 = _solvable_task()
    t1.generated_solution = _multi_step_solution()
    t1.match_score_LLM = ScoreValues.MATCH
    t1.match_reasoning_LLM = "Candidate matches the reference solution exactly."

    t2 = _unsolvable_task()
    t2.generated_solution = Solution(steps=[Step(function_name="load_geodata", arguments={})])
    t2.match_score_LLM = ScoreValues.NO_MATCH
    t2.match_reasoning_LLM = "Candidate did not reject the task."
    t2.match_score_Human = ScoreValues.NO_MATCH
    t2.match_reasoning_Human = "Agreed with the LLM judge."

    t3 = Task(task_ID="TASK_003", task_text="unscored task")  # match_score_LLM stays None

    return TaskSet(tasks=[t1, t2, t3])


def test_results_to_openeval_produces_valid_result_set():
    result_set = results_to_openeval(_scored_task_set(), run_id="run_1", started_at="2026-08-31T00:00:00Z")
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_results_to_openeval_skips_unscored_tasks():
    result_set = results_to_openeval(_scored_task_set(), run_id="run_1", started_at="2026-08-31T00:00:00Z")
    ids = {r["test_case_id"] for r in result_set["results"]}
    assert ids == {"TASK_001", "TASK_002"}  # TASK_003 has no match_score_LLM


def test_results_to_openeval_normalizes_score_and_preserves_raw():
    result_set = results_to_openeval(_scored_task_set(), run_id="run_1", started_at="2026-08-31T00:00:00Z")
    by_id = {r["test_case_id"]: r for r in result_set["results"]}

    match_result = by_id["TASK_001"]
    gr = match_result["grader_results"][0]
    assert gr["score"] == 1.0  # ScoreValues.MATCH (2) / 2.0
    assert gr["passed"] is True
    assert gr["metadata"]["raw_score_0_1_2"] == 2
    assert match_result["passed"] is True

    no_match_result = by_id["TASK_002"]
    gr2 = no_match_result["grader_results"][0]
    assert gr2["score"] == 0.0
    assert gr2["passed"] is False
    assert gr2["metadata"]["raw_score_0_1_2"] == 0
    assert gr2["metadata"]["human_score"] == 0
    assert gr2["metadata"]["human_reasoning"] == "Agreed with the LLM judge."


def test_results_to_openeval_actual_output_renders_generated_solution():
    result_set = results_to_openeval(_scored_task_set(), run_id="run_1", started_at="2026-08-31T00:00:00Z")
    match_result = next(r for r in result_set["results"] if r["test_case_id"] == "TASK_001")
    assert "load_geodata(" in match_result["actual_output"]
    assert "calculate_line_lengths(" in match_result["actual_output"]


def test_results_to_openeval_includes_citation():
    result_set = results_to_openeval(_scored_task_set(), run_id="run_1", started_at="2026-08-31T00:00:00Z")
    assert result_set["metadata"]["citation"]["doi"] == "10.1145/3764915.3770721"


def test_results_to_openeval_completed_at_optional():
    result_set = results_to_openeval(
        _scored_task_set(), run_id="run_1", started_at="2026-08-31T00:00:00Z"
    )
    assert "completed_at" not in result_set

    result_set2 = results_to_openeval(
        _scored_task_set(), run_id="run_1", started_at="2026-08-31T00:00:00Z", completed_at="2026-08-31T00:05:00Z"
    )
    assert result_set2["completed_at"] == "2026-08-31T00:05:00Z"


# ---------------------------------------------------------------------------
# from_openeval (round trip)
# ---------------------------------------------------------------------------

def test_from_openeval_round_trip_reconstructs_real_geobenchx_task():
    original = _solvable_task()
    suite = to_openeval(TaskSet(tasks=[original]))

    reconstructed_dicts = from_openeval(suite)
    assert len(reconstructed_dicts) == 1

    # This is the real assertion: the dict really constructs a valid,
    # equivalent geobenchx.dataclasses.Task -- not just a dict that looks
    # plausible.
    rebuilt = Task(**reconstructed_dicts[0])

    assert rebuilt.task_ID == original.task_ID
    assert rebuilt.task_text == original.task_text
    assert set(rebuilt.task_labels) == set(original.task_labels)
    assert rebuilt.reference_solution_description == original.reference_solution_description

    assert len(rebuilt.reference_solutions) == len(original.reference_solutions)
    rebuilt_steps = rebuilt.reference_solutions[0].steps
    original_steps = original.reference_solutions[0].steps
    assert [s.function_name for s in rebuilt_steps] == [s.function_name for s in original_steps]
    assert [s.arguments for s in rebuilt_steps] == [s.arguments for s in original_steps]
    assert [s.comment for s in rebuilt_steps] == [s.comment for s in original_steps]


def test_from_openeval_round_trip_preserves_reject_task_only_solution():
    original = _unsolvable_task()
    suite = to_openeval(TaskSet(tasks=[original]))
    rebuilt = Task(**from_openeval(suite)[0])

    assert len(rebuilt.reference_solutions) == 1
    assert [s.function_name for s in rebuilt.reference_solutions[0].steps] == ["reject_task"]


def test_from_openeval_handles_suite_with_no_metadata():
    # A TestCase without the metadata.reference_solutions this adapter writes
    # (e.g. authored by hand, or by a different adapter) should still degrade
    # gracefully rather than raising.
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
    }
    tasks = from_openeval(suite)
    assert tasks == [
        {
            "task_ID": "tc1",
            "task_text": "hi",
            "task_labels": [],
            "reference_solution_description": None,
            "reference_solutions": [],
        }
    ]
    # And it still constructs a valid (if information-poor) real Task.
    Task(**tasks[0])
