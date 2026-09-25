from openeval.validate import validate_suite, validate_result_set

from mmmu_openeval_adapter import to_openeval, from_openeval, result_to_openeval


MULTI_CHOICE_SAMPLE = {
    "id": "validation_Accounting_1",
    "question": "What is the value of X in the balance sheet?",
    "options": "['10', '20', '30', '40']",
    "answer": "B",
    "question_type": "multiple-choice",
}

OPEN_SAMPLE = {
    "id": "validation_Math_7",
    "question": "What is the area of the shaded region?",
    "answer": "3.14",
    "question_type": "open",
}

OPEN_SAMPLE_LIST_ANSWER = {
    "id": "validation_Physics_2",
    "question": "What is the speed in m/s?",
    "answer": ["9.8", "9.81"],
    "question_type": "open",
}


def test_to_openeval_multiple_choice():
    suite = to_openeval([MULTI_CHOICE_SAMPLE], suite_id="mmmu_validation")

    assert suite["id"] == "mmmu_validation"
    tc = suite["test_cases"][0]
    assert tc["id"] == "validation_Accounting_1"
    assert tc["expected_output"] == "B"
    assert "(A) 10" in tc["input"]
    assert "(B) 20" in tc["input"]
    assert tc["graders"] == ["gr_mmmu_multi_choice"]
    assert tc["metadata"]["mmmu"]["question_type"] == "multiple-choice"
    assert tc["metadata"]["mmmu"]["category"] == "Accounting"
    assert tc["metadata"]["mmmu"]["domain"] == "Business"
    assert tc["metadata"]["mmmu"]["options"] == ["10", "20", "30", "40"]
    assert tc["metadata"]["mmmu"]["index2ans"] == {"A": "10", "B": "20", "C": "30", "D": "40"}

    grader_ids = {g["id"] for g in suite["graders"]}
    assert "gr_mmmu_multi_choice" in grader_ids


def test_to_openeval_open_question():
    suite = to_openeval([OPEN_SAMPLE], suite_id="mmmu_validation")
    tc = suite["test_cases"][0]
    assert tc["input"] == "What is the area of the shaded region?"
    assert tc["expected_output"] == "3.14"
    assert tc["graders"] == ["gr_mmmu_open"]
    assert tc["metadata"]["mmmu"]["category"] == "Math"
    assert tc["metadata"]["mmmu"]["domain"] == "Science"

    grader_types = {g["id"]: g["type"] for g in suite["graders"]}
    assert grader_types["gr_mmmu_open"] == "custom"


def test_to_openeval_open_question_list_answer():
    suite = to_openeval([OPEN_SAMPLE_LIST_ANSWER], suite_id="mmmu_validation")
    tc = suite["test_cases"][0]
    # First acceptable answer is used as the single expected_output ...
    assert tc["expected_output"] == "9.8"
    # ... but the full acceptable set is preserved, nothing silently dropped.
    assert tc["metadata"]["mmmu"]["acceptable_answers"] == ["9.8", "9.81"]


def test_to_openeval_has_image_flag_without_binary():
    sample = dict(MULTI_CHOICE_SAMPLE, image_1="/tmp/some_image.png")
    suite = to_openeval([sample], suite_id="mmmu_validation")
    meta = suite["test_cases"][0]["metadata"]["mmmu"]
    assert meta["has_image"] is True
    assert meta["image_field"] == "image_1"
    assert meta["image"] == "/tmp/some_image.png"


def test_to_openeval_validates_against_evalport_spec():
    suite = to_openeval([MULTI_CHOICE_SAMPLE, OPEN_SAMPLE], suite_id="mmmu_validation")
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_to_openeval_rejects_empty_samples():
    try:
        to_openeval([])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_from_openeval_round_trip():
    suite = to_openeval([MULTI_CHOICE_SAMPLE, OPEN_SAMPLE], suite_id="mmmu_validation")
    recovered = from_openeval(suite)

    assert len(recovered) == 2
    mc = next(s for s in recovered if s["id"] == "validation_Accounting_1")
    assert mc["question"] == "What is the value of X in the balance sheet?"
    assert mc["options"] == ["10", "20", "30", "40"]
    assert mc["answer"] == "B"
    assert mc["question_type"] == "multiple-choice"

    open_q = next(s for s in recovered if s["id"] == "validation_Math_7")
    assert open_q["question"] == "What is the area of the shaded region?"
    assert open_q["answer"] == "3.14"
    assert open_q["question_type"] == "open"


def test_from_openeval_skips_foreign_test_cases():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
    }
    assert from_openeval(suite) == []


def test_result_to_openeval_matches_main_eval_only_shape():
    exampels_to_eval = [
        {
            "id": "validation_Accounting_1",
            "question_type": "multiple-choice",
            "answer": "B",
            "parsed_pred": "B",
        },
        {
            "id": "validation_Math_7",
            "question_type": "open",
            "answer": "3.14",
            "parsed_pred": [3.14, "3.14"],
        },
        {
            "id": "validation_Math_8",
            "question_type": "open",
            "answer": "5",
            "parsed_pred": [4.0, "4"],
        },
    ]
    judge_dict = {
        "validation_Accounting_1": "Correct",
        "validation_Math_7": "Correct",
        "validation_Math_8": "Wrong",
    }
    # Mirrors main_eval_only.py's own per-category/per-domain rollup shape.
    printable_results = {
        "Overall-Business": {"num": 1, "acc": 1.0},
        "Accounting": {"num": 1, "acc": 1.0},
        "Overall-Science": {"num": 2, "acc": 0.5},
        "Math": {"num": 2, "acc": 0.5},
        "Overall": {"num": 3, "acc": round(2 / 3, 3)},
    }

    result_set = result_to_openeval(
        exampels_to_eval,
        judge_dict,
        printable_results,
        suite_id="mmmu_validation",
        run_id="run_2026_09_24",
        started_at="2026-09-24T00:00:00Z",
        completed_at="2026-09-24T00:05:00Z",
    )

    assert result_set["suite_id"] == "mmmu_validation"
    assert result_set["run_id"] == "run_2026_09_24"
    assert len(result_set["results"]) == 3

    acc = next(r for r in result_set["results"] if r["test_case_id"] == "validation_Accounting_1")
    assert acc["passed"] is True
    assert acc["grader_results"][0]["grader_id"] == "gr_mmmu_multi_choice"
    assert acc["grader_results"][0]["type"] == "exact_match"
    assert acc["grader_results"][0]["score"] == 1.0

    wrong = next(r for r in result_set["results"] if r["test_case_id"] == "validation_Math_8")
    assert wrong["passed"] is False
    assert wrong["grader_results"][0]["grader_id"] == "gr_mmmu_open"
    assert wrong["grader_results"][0]["type"] == "custom"
    assert wrong["grader_results"][0]["score"] == 0.0
    assert wrong["actual_output"] == "4.0, 4"

    # printable_results survives verbatim as ResultSet.summary -- the
    # per-discipline breakdown is the entire point of this adapter.
    assert result_set["summary"] == printable_results
    assert result_set["summary"]["Overall"]["acc"] == round(2 / 3, 3)


def test_result_to_openeval_validates_against_evalport_spec():
    exampels_to_eval = [
        {"id": "validation_Accounting_1", "question_type": "multiple-choice", "answer": "B", "parsed_pred": "B"},
    ]
    judge_dict = {"validation_Accounting_1": "Correct"}
    printable_results = {"Overall": {"num": 1, "acc": 1.0}}

    result_set = result_to_openeval(
        exampels_to_eval,
        judge_dict,
        printable_results,
        suite_id="mmmu_validation",
        run_id="run1",
        started_at="2026-09-24T00:00:00Z",
    )
    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors


def test_result_to_openeval_rejects_empty_input():
    try:
        result_to_openeval([], {}, {}, suite_id="s", run_id="r", started_at="2026-09-24T00:00:00Z")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_result_to_openeval_missing_judge_verdict_raises():
    exampels_to_eval = [
        {"id": "validation_Accounting_1", "question_type": "multiple-choice", "answer": "B", "parsed_pred": "B"},
    ]
    try:
        result_to_openeval(
            exampels_to_eval, {}, {}, suite_id="s", run_id="r", started_at="2026-09-24T00:00:00Z"
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_full_round_trip_suite_and_results_share_ids():
    """The suite built by to_openeval() and the ResultSet built by
    result_to_openeval() must reference the same TestCase ids, since a real
    consumer joins them to compute pass rates per test case."""
    suite = to_openeval([MULTI_CHOICE_SAMPLE, OPEN_SAMPLE], suite_id="mmmu_validation")
    suite_ids = {tc["id"] for tc in suite["test_cases"]}

    exampels_to_eval = [
        {"id": "validation_Accounting_1", "question_type": "multiple-choice", "answer": "B", "parsed_pred": "B"},
        {"id": "validation_Math_7", "question_type": "open", "answer": "3.14", "parsed_pred": [3.14]},
    ]
    judge_dict = {"validation_Accounting_1": "Correct", "validation_Math_7": "Correct"}
    printable_results = {"Overall": {"num": 2, "acc": 1.0}}

    result_set = result_to_openeval(
        exampels_to_eval,
        judge_dict,
        printable_results,
        suite_id="mmmu_validation",
        run_id="run1",
        started_at="2026-09-24T00:00:00Z",
    )
    result_ids = {r["test_case_id"] for r in result_set["results"]}

    assert result_ids == suite_ids
    assert validate_suite(suite).valid
    assert validate_result_set(result_set).valid
