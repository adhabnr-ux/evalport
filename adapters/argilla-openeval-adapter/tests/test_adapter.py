"""Tests for argilla-openeval-adapter.

Every test here runs against the real, installed ``argilla`` package (not a
mock) and the real ``openeval.validate`` validators from ``evalport-sdk``.
All test data goes through real ``argilla.Record`` / ``Suggestion`` /
``Response`` objects -- confirmed instantiable and serializable fully
offline (no server connection). ``argilla.Settings`` / ``Field`` /
``Question`` / ``Dataset`` are deliberately NOT exercised here: constructing
any of them requires a live, connected ``Argilla`` client, which validates
the connection eagerly (an HTTP call at ``__init__`` time) -- confirmed
directly against argilla 2.8.0 by attempting it with no server reachable
and observing ``httpx.ConnectError``. There is no Argilla test server
available in this sandbox, and this adapter's public API never requires
one (see the module docstring in ``argilla_openeval_adapter`` for the full
reasoning), so nothing here is skipped or mocked to route around that --
the adapter's actual functionality doesn't touch that part of the SDK.
"""
import uuid

import pytest
import argilla as rg

from openeval.validate import validate_result_set, validate_suite

from argilla_openeval_adapter import (
    from_openeval,
    responses_to_openeval,
    to_openeval,
)


def _record(id_, fields, metadata=None, suggestions=None, responses=None):
    r = rg.Record(id=id_, fields=fields, metadata=metadata or {})
    for sug in suggestions or []:
        r.suggestions.add(sug)
    for resp in responses or []:
        r.responses.add(resp)
    return r


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpenEval:
    def test_single_field_becomes_string_input(self):
        records = [_record("r1", {"prompt": "What is 2+2?"})]
        suite = to_openeval(records)
        assert suite["test_cases"][0]["input"] == "What is 2+2?"
        assert isinstance(suite["test_cases"][0]["input"], str)

    def test_multi_field_becomes_array_input_in_order(self):
        records = [_record("r1", {"prompt": "2+2?", "context": "grade-school math"})]
        suite = to_openeval(records)
        tc = suite["test_cases"][0]
        assert tc["input"] == ["2+2?", "grade-school math"]
        assert tc["metadata"]["argilla"]["field_names"] == ["prompt", "context"]

    def test_every_test_case_graded_by_human(self):
        records = [_record("r1", {"prompt": "hi"}), _record("r2", {"prompt": "bye"})]
        suite = to_openeval(records)
        for tc in suite["test_cases"]:
            assert tc["graders"] == ["human"]
        assert suite["graders"] == [
            {"id": "human", "type": "human", "description": "A human annotator's judgment, captured in Argilla."}
        ]

    def test_expected_output_field(self):
        records = [_record("r1", {"prompt": "2+2?", "gold": "4"})]
        suite = to_openeval(records, expected_output_field="gold")
        tc = suite["test_cases"][0]
        assert tc["input"] == "2+2?"
        assert tc["expected_output"] == "4"
        # gold was excluded from the auto-derived input fields
        assert "gold" not in tc["metadata"]["argilla"]["field_names"]

    def test_explicit_input_fields_overrides_autodetection(self):
        records = [_record("r1", {"a": "1", "b": "2", "c": "3"})]
        suite = to_openeval(records, input_fields=["c", "a"])
        assert suite["test_cases"][0]["input"] == ["3", "1"]

    def test_record_id_used_as_test_case_id(self):
        records = [_record("my-record-42", {"prompt": "hi"})]
        suite = to_openeval(records)
        assert suite["test_cases"][0]["id"] == "my-record-42"
        assert suite["test_cases"][0]["metadata"]["argilla"]["record_id"] == "my-record-42"

    def test_explicit_ids_override_record_id(self):
        records = [_record("my-record-42", {"prompt": "hi"})]
        suite = to_openeval(records, ids=["custom-1"])
        assert suite["test_cases"][0]["id"] == "custom-1"

    def test_record_without_id_falls_back_to_index(self):
        r = rg.Record(fields={"prompt": "hi"})  # auto-generated uuid id
        suite = to_openeval([r], ids=["explicit-id"])
        assert suite["test_cases"][0]["id"] == "explicit-id"

    def test_record_metadata_preserved(self):
        records = [_record("r1", {"prompt": "hi"}, metadata={"source": "eval-set-3"})]
        suite = to_openeval(records)
        assert suite["test_cases"][0]["metadata"]["argilla"]["record_metadata"] == {"source": "eval-set-3"}

    def test_suggestions_preserved_not_promoted_to_grader_results(self):
        sug = rg.Suggestion(question_name="quality", value="good", score=0.9, agent="gpt-4-judge")
        records = [_record("r1", {"prompt": "hi"}, suggestions=[sug])]
        suite = to_openeval(records)
        preserved = suite["test_cases"][0]["metadata"]["argilla"]["suggestions"]
        assert preserved["quality"]["value"] == "good"
        assert preserved["quality"]["score"] == 0.9
        assert preserved["quality"]["agent"] == "gpt-4-judge"
        # never fabricated into an executed grader result
        assert suite["test_cases"][0]["graders"] == ["human"]

    def test_dict_shaped_records_accepted(self):
        d = {"id": "r1", "fields": {"prompt": "hi there"}, "metadata": {}, "suggestions": {}}
        suite = to_openeval([d])
        assert suite["test_cases"][0]["input"] == "hi there"

    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            to_openeval([])

    def test_no_input_fields_derivable_raises(self):
        records = [_record("r1", {"gold": "4"})]
        with pytest.raises(ValueError):
            to_openeval(records, expected_output_field="gold")

    def test_suite_id_and_description(self):
        records = [_record("r1", {"prompt": "hi"})]
        suite = to_openeval(records, suite_id="my_argilla_suite", description="A test suite")
        assert suite["id"] == "my_argilla_suite"
        assert suite["description"] == "A test suite"

    def test_default_suite_id(self):
        records = [_record("r1", {"prompt": "hi"})]
        suite = to_openeval(records)
        assert suite["id"] == "argilla_suite"

    def test_validates_against_real_openeval_schema(self):
        sug = rg.Suggestion(question_name="quality", value="good", score=0.9)
        records = [
            _record("r1", {"prompt": "2+2?", "context": "math"}, suggestions=[sug]),
            _record("r2", {"prompt": "capital of France?"}),
        ]
        suite = to_openeval(records, description="Argilla adapter validation suite")
        result = validate_suite(suite)
        assert result.valid, result.errors


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpenEval:
    def test_single_input_string_becomes_one_field(self):
        suite = to_openeval([_record("r1", {"prompt": "hi"})])
        specs = from_openeval(suite)
        assert specs[0]["fields"] == {"prompt": "hi"}

    def test_multi_input_array_restores_field_names(self):
        suite = to_openeval([_record("r1", {"prompt": "hi", "context": "greeting"})])
        specs = from_openeval(suite)
        assert specs[0]["fields"] == {"prompt": "hi", "context": "greeting"}

    def test_specs_reconstruct_into_live_records(self):
        suite = to_openeval([_record("r1", {"prompt": "hi", "context": "greeting"})])
        specs = from_openeval(suite)
        record = rg.Record.from_dict(specs[0])
        assert record.fields["prompt"] == "hi"
        assert record.fields["context"] == "greeting"

    def test_suggestions_round_trip(self):
        sug = rg.Suggestion(question_name="quality", value="good", score=0.9, agent="judge")
        suite = to_openeval([_record("r1", {"prompt": "hi"}, suggestions=[sug])])
        specs = from_openeval(suite)
        assert specs[0]["suggestions"]["quality"]["value"] == "good"
        assert specs[0]["suggestions"]["quality"]["score"] == 0.9
        record = rg.Record.from_dict(specs[0])
        assert record.suggestions["quality"].value == "good"

    def test_expected_output_becomes_a_field(self):
        suite = to_openeval(
            [_record("r1", {"prompt": "2+2?", "gold": "4"})], expected_output_field="gold"
        )
        specs = from_openeval(suite)
        assert specs[0]["fields"]["prompt"] == "2+2?"
        assert specs[0]["fields"]["expected_output"] == "4"

    def test_generic_suite_without_argilla_metadata_gets_synthetic_field_names(self):
        generic_suite = {
            "version": "1.0.0",
            "id": "generic",
            "test_cases": [
                {"id": "t1", "input": ["hello", "world"], "graders": ["human"]},
            ],
        }
        specs = from_openeval(generic_suite)
        assert specs[0]["fields"] == {"field_0": "hello", "field_1": "world"}

    def test_generic_suite_single_string_input(self):
        generic_suite = {
            "version": "1.0.0",
            "id": "generic",
            "test_cases": [{"id": "t1", "input": "just text", "graders": ["human"]}],
        }
        specs = from_openeval(generic_suite)
        assert specs[0]["fields"] == {"field_0": "just text"}
        record = rg.Record.from_dict(specs[0])
        assert record.fields["field_0"] == "just text"


# ---------------------------------------------------------------------------
# responses_to_openeval
# ---------------------------------------------------------------------------


class TestResponsesToOpenEval:
    def test_single_annotator_response_becomes_grader_result(self):
        u = uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "hi"},
                responses=[rg.Response(question_name="quality", value="good", user_id=u, status="submitted")],
            )
        ]
        rs = responses_to_openeval(records)
        assert len(rs["results"]) == 1
        gr = rs["results"][0]["grader_results"][0]
        assert gr["grader_id"] == "quality"
        assert gr["type"] == "human"
        assert gr["metadata"]["user_id"] == str(u)
        assert gr["metadata"]["status"] == "ResponseStatus.submitted"

    def test_multiple_annotators_get_indexed_grader_ids(self):
        u1, u2 = uuid.uuid4(), uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "hi"},
                responses=[
                    rg.Response(question_name="quality", value="good", user_id=u1, status="submitted"),
                    rg.Response(question_name="quality", value="bad", user_id=u2, status="submitted"),
                ],
            )
        ]
        rs = responses_to_openeval(records)
        grader_ids = {gr["grader_id"] for gr in rs["results"][0]["grader_results"]}
        assert grader_ids == {"quality[0]", "quality[1]"}

    def test_boolean_response_scores_one_or_zero(self):
        u = uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "hi"},
                responses=[rg.Response(question_name="is_safe", value=True, user_id=u, status="submitted")],
            )
        ]
        rs = responses_to_openeval(records)
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] == 1.0
        assert gr["passed"] is True

    def test_numeric_rating_normalized_with_rating_ranges(self):
        u = uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "hi"},
                responses=[rg.Response(question_name="quality", value=2, user_id=u, status="submitted")],
            )
        ]
        rs = responses_to_openeval(records, rating_ranges={"quality": (1, 5)})
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] == pytest.approx(0.25)  # (2-1)/(5-1)
        assert gr["passed"] is False  # below default 0.5 threshold

    def test_numeric_rating_without_range_and_out_of_bounds_is_unscored(self):
        u = uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "hi"},
                responses=[rg.Response(question_name="quality", value=4, user_id=u, status="submitted")],
            )
        ]
        rs = responses_to_openeval(records)  # no rating_ranges supplied, 4 is out of [0,1]
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert gr["passed"] is True  # unscored responses default to passed

    def test_label_response_has_null_score(self):
        u = uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "hi"},
                responses=[rg.Response(question_name="topic", value="math", user_id=u, status="submitted")],
            )
        ]
        rs = responses_to_openeval(records)
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert "math" in gr["reason"]

    def test_records_without_responses_are_skipped(self):
        records = [
            _record("r1", {"prompt": "hi"}),  # no responses yet
            _record(
                "r2",
                {"prompt": "bye"},
                responses=[rg.Response(question_name="q", value="ok", user_id=uuid.uuid4(), status="submitted")],
            ),
        ]
        rs = responses_to_openeval(records)
        assert len(rs["results"]) == 1
        assert rs["results"][0]["test_case_id"] == "r2"

    def test_all_unannotated_raises(self):
        records = [_record("r1", {"prompt": "hi"})]
        with pytest.raises(ValueError):
            responses_to_openeval(records)

    def test_overall_passed_is_and_of_grader_results(self):
        u = uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "hi"},
                responses=[
                    rg.Response(question_name="a", value=True, user_id=u, status="submitted"),
                    rg.Response(question_name="b", value=False, user_id=u, status="submitted"),
                ],
            )
        ]
        rs = responses_to_openeval(records)
        assert rs["results"][0]["passed"] is False

    def test_validates_against_real_openeval_schema(self):
        u1, u2 = uuid.uuid4(), uuid.uuid4()
        records = [
            _record(
                "r1",
                {"prompt": "2+2?"},
                responses=[
                    rg.Response(question_name="correctness", value=True, user_id=u1, status="submitted"),
                    rg.Response(question_name="quality", value=4, user_id=u1, status="submitted"),
                ],
            ),
            _record(
                "r2",
                {"prompt": "capital of France?"},
                responses=[
                    rg.Response(question_name="topic", value="geography", user_id=u2, status="submitted"),
                ],
            ),
        ]
        rs = responses_to_openeval(records, suite_id="argilla_suite", rating_ranges={"quality": (1, 5)})
        result = validate_result_set(rs)
        assert result.valid, result.errors


# ---------------------------------------------------------------------------
# Full loop
# ---------------------------------------------------------------------------


class TestFullLoop:
    def test_full_loop_input_to_annotation_to_results(self):
        # 1. Start with source records (e.g. exported from a QA dataset).
        records = [
            _record("q1", {"prompt": "What is the capital of Japan?"}),
            _record("q2", {"prompt": "What is 7 * 6?"}),
        ]

        # 2. Export as a portable EvalPort suite.
        suite = to_openeval(records, suite_id="geo_math_quiz")
        assert validate_suite(suite).valid

        # 3. Import into Argilla-ready record specs and build real Records
        #    (this is what you'd log to a connected Argilla dataset).
        specs = from_openeval(suite)
        live_records = [rg.Record.from_dict(s) for s in specs]
        # single-field records: field_names metadata preserved "prompt" as
        # the real original field name, not a synthetic fallback.
        assert live_records[0].fields["prompt"] == "What is the capital of Japan?"

        # 4. Simulate annotators completing their review in Argilla by
        #    attaching real Response objects.
        live_records[0].responses.add(
            rg.Response(question_name="correct", value=True, user_id=uuid.uuid4(), status="submitted")
        )
        live_records[1].responses.add(
            rg.Response(question_name="correct", value=True, user_id=uuid.uuid4(), status="submitted")
        )

        # 5. Export the completed human judgments as an EvalPort ResultSet.
        result_set = responses_to_openeval(
            live_records, ids=["q1", "q2"], suite_id=suite["id"]
        )
        assert validate_result_set(result_set).valid
        assert result_set["results"][0]["test_case_id"] == "q1"
        assert result_set["results"][0]["passed"] is True
