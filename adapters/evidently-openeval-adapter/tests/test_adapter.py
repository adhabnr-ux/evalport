"""Tests for evidently_openeval_adapter.

Every test runs against the real ``evidently`` package (real
``evidently.Dataset``/``ExactMatch``/``Contains``/``TextLength``
descriptors) and the real ``evalport-sdk`` validators
(``openeval.validate.validate_suite`` / ``validate_result_set``) -- nothing
here is mocked.
"""
from __future__ import annotations

import pandas as pd
import pytest
from evidently import DataDefinition, Dataset
from evidently.descriptors import Contains, ExactMatch, TextLength
from openeval.validate import validate_result_set, validate_suite

from evidently_openeval_adapter import (
    evaluation_result_to_openeval,
    from_openeval,
    to_openeval,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "question": [
                "What is the capital of France?",
                "What is the capital of Japan?",
                "What is the capital of Germany?",
            ],
            "expected": ["Paris", "Tokyo", "Berlin"],
            "answer": ["Paris", "Tokyo", "Munich"],
        }
    )


# ---------------------------------------------------------------------------
# to_openeval
# ---------------------------------------------------------------------------


class TestToOpenEval:
    def test_basic_shape_and_validates(self, df):
        suite = to_openeval(
            df,
            input_columns=["question"],
            expected_column="expected",
            graders=["exact_match"],
            descriptor_types={"exact_match": "ExactMatch"},
        )
        result = validate_suite(suite)
        assert result.valid, result.errors
        assert len(suite["test_cases"]) == 3
        assert suite["id"] == "evidently_suite"

    def test_default_suite_id_and_placeholder_grader(self, df):
        suite = to_openeval(df, input_columns=["question"])
        assert suite["graders"] == [
            {
                "id": "evidently_descriptor",
                "type": "custom",
                "params": {"handler": "evidently_descriptor"},
                "description": (
                    "Placeholder for the Evidently descriptor aliased "
                    "'evidently_descriptor' -- the caller must run the "
                    "actual evidently.Dataset evaluation and supply its "
                    "column values via evaluation_result_to_openeval(), "
                    "rather than this module fabricating a fake "
                    "implementation."
                ),
            }
        ]
        assert validate_suite(suite).valid

    def test_custom_suite_id_and_description(self, df):
        suite = to_openeval(
            df, input_columns=["question"], suite_id="my_qa_suite", description="QA eval"
        )
        assert suite["id"] == "my_qa_suite"
        assert suite["description"] == "QA eval"

    def test_custom_ids(self, df):
        suite = to_openeval(df, input_columns=["question"], ids=["fr", "jp", "de"])
        assert [tc["id"] for tc in suite["test_cases"]] == ["fr", "jp", "de"]
        assert validate_suite(suite).valid

    def test_multiple_input_columns_flattened(self, df):
        suite = to_openeval(df, input_columns=["question", "answer"])
        tc0 = suite["test_cases"][0]
        assert tc0["input"] == [
            "question: What is the capital of France?",
            "answer: Paris",
        ]
        assert validate_suite(suite).valid

    def test_expected_column_becomes_expected_output(self, df):
        suite = to_openeval(df, input_columns=["question"], expected_column="expected")
        assert suite["test_cases"][0]["expected_output"] == "Paris"
        assert suite["test_cases"][2]["expected_output"] == "Berlin"

    def test_missing_expected_column_omits_field(self, df):
        suite = to_openeval(df, input_columns=["question"], expected_column="nonexistent")
        assert "expected_output" not in suite["test_cases"][0]

    def test_metadata_preserves_full_row_as_native_types(self, df):
        suite = to_openeval(df, input_columns=["question"])
        row_meta = suite["test_cases"][1]["metadata"]["evidently"]["columns"]
        assert row_meta == {
            "question": "What is the capital of Japan?",
            "expected": "Tokyo",
            "answer": "Tokyo",
        }
        for v in row_meta.values():
            assert type(v) is str  # native Python, not a numpy/pandas scalar

    def test_graders_list_infers_exact_match_type(self, df):
        suite = to_openeval(
            df,
            input_columns=["question"],
            graders=["exact_match"],
            descriptor_types={"exact_match": "ExactMatch"},
        )
        grader = suite["graders"][0]
        assert grader["id"] == "exact_match"
        assert grader["type"] == "exact_match"
        assert "params" not in grader
        for tc in suite["test_cases"]:
            assert tc["graders"] == ["exact_match"]
        assert validate_suite(suite).valid

    def test_graders_without_descriptor_types_falls_back_to_custom(self, df):
        suite = to_openeval(df, input_columns=["question"], graders=["exact_match"])
        grader = suite["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"] == {"handler": "exact_match"}
        assert validate_suite(suite).valid

    def test_unrecognized_descriptor_class_falls_back_to_custom(self, df):
        suite = to_openeval(
            df,
            input_columns=["question"],
            graders=["sentiment"],
            descriptor_types={"sentiment": "Sentiment"},
        )
        grader = suite["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"] == {"handler": "sentiment"}
        assert validate_suite(suite).valid

    def test_multiple_graders(self, df):
        suite = to_openeval(
            df,
            input_columns=["question"],
            graders=["exact_match", "answer_length"],
            descriptor_types={"exact_match": "ExactMatch"},
        )
        grader_ids = [g["id"] for g in suite["graders"]]
        assert grader_ids == ["exact_match", "answer_length"]
        assert suite["graders"][0]["type"] == "exact_match"
        assert suite["graders"][1]["type"] == "custom"
        for tc in suite["test_cases"]:
            assert tc["graders"] == ["exact_match", "answer_length"]
        assert validate_suite(suite).valid

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="df is empty"):
            to_openeval(pd.DataFrame(), input_columns=["x"])

    def test_empty_input_columns_raises(self, df):
        with pytest.raises(ValueError, match="input_columns is empty"):
            to_openeval(df, input_columns=[])

    def test_missing_input_column_raises(self, df):
        with pytest.raises(ValueError, match="not present in df"):
            to_openeval(df, input_columns=["nonexistent_column"])

    def test_mismatched_ids_length_raises(self, df):
        with pytest.raises(ValueError, match="ids has length"):
            to_openeval(df, input_columns=["question"], ids=["only_one"])


# ---------------------------------------------------------------------------
# from_openeval
# ---------------------------------------------------------------------------


class TestFromOpenEval:
    def test_lossless_round_trip(self, df):
        suite = to_openeval(
            df,
            input_columns=["question"],
            expected_column="expected",
            graders=["exact_match"],
        )
        result_df = from_openeval(suite)
        non_id = result_df.drop(columns=["id"])
        pd.testing.assert_frame_equal(
            non_id.reset_index(drop=True).sort_index(axis=1),
            df.reset_index(drop=True).sort_index(axis=1),
        )
        assert list(result_df["id"]) == ["evidently_tc_0", "evidently_tc_1", "evidently_tc_2"]

    def test_lossless_round_trip_preserves_row_order(self, df):
        suite = to_openeval(df, input_columns=["question"], ids=["a", "b", "c"])
        result_df = from_openeval(suite)
        assert list(result_df["question"]) == list(df["question"])
        assert list(result_df["id"]) == ["a", "b", "c"]

    def test_id_column_not_overwritten_when_already_real_data(self):
        df_with_real_id = pd.DataFrame(
            {"question": ["q1", "q2"], "id": ["real-id-1", "real-id-2"]}
        )
        suite = to_openeval(df_with_real_id, input_columns=["question"], ids=["tc_a", "tc_b"])
        result_df = from_openeval(suite)
        assert list(result_df["id"]) == ["real-id-1", "real-id-2"]

    def test_foreign_suite_positional_naming(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [
                {"id": "tc1", "input": ["hello", "world"], "graders": ["g1"], "expected_output": "hi"},
                {"id": "tc2", "input": ["foo", "bar"], "graders": ["g1"], "expected_output": "baz"},
            ],
        }
        result_df = from_openeval(suite)
        expected = pd.DataFrame(
            {
                "expected_output": ["hi", "baz"],
                "id": ["tc1", "tc2"],
                "input_1": ["hello", "foo"],
                "input_2": ["world", "bar"],
            }
        )
        pd.testing.assert_frame_equal(
            result_df.sort_index(axis=1).reset_index(drop=True),
            expected.sort_index(axis=1).reset_index(drop=True),
        )

    def test_foreign_suite_explicit_input_columns(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [
                {"id": "tc1", "input": ["hello"], "graders": ["g1"]},
                {"id": "tc2", "input": ["foo"], "graders": ["g1"]},
            ],
        }
        result_df = from_openeval(suite, input_columns=["question"])
        assert list(result_df["question"]) == ["hello", "foo"]
        assert list(result_df["id"]) == ["tc1", "tc2"]

    def test_mismatched_input_columns_length_raises(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [{"id": "tc1", "input": ["a", "b"], "graders": ["g1"]}],
        }
        with pytest.raises(ValueError, match="input entries but input_columns has"):
            from_openeval(suite, input_columns=["only_one_name"])

    def test_empty_test_cases_raises(self):
        with pytest.raises(ValueError, match="no test_cases"):
            from_openeval({"version": "1.0.0", "id": "s", "graders": [], "test_cases": []})

    def test_inconsistent_columns_raises(self):
        suite = {
            "version": "1.0.0",
            "id": "foreign_suite",
            "graders": [{"id": "g1", "type": "custom", "params": {"handler": "g1"}}],
            "test_cases": [
                {
                    "id": "tc1",
                    "input": ["x"],
                    "graders": ["g1"],
                    "metadata": {"evidently": {"columns": {"a": 1, "b": 2}}},
                },
                {
                    "id": "tc2",
                    "input": ["y"],
                    "graders": ["g1"],
                    "metadata": {"evidently": {"columns": {"a": 3, "c": 4}}},
                },
            ],
        }
        with pytest.raises(ValueError, match="inconsistent column names"):
            from_openeval(suite)


# ---------------------------------------------------------------------------
# evaluation_result_to_openeval
# ---------------------------------------------------------------------------


class TestEvaluationResultToOpenEval:
    def test_real_exact_match_descriptor_round_trip(self, df):
        suite = to_openeval(
            df,
            input_columns=["question"],
            expected_column="expected",
            graders=["exact_match"],
            descriptor_types={"exact_match": "ExactMatch"},
            ids=["fr", "jp", "de"],
        )
        result_df = from_openeval(suite)

        dataset = Dataset.from_pandas(
            result_df,
            data_definition=DataDefinition(),
            descriptors=[ExactMatch(columns=["expected", "answer"], alias="exact_match")],
        )

        result_set = evaluation_result_to_openeval(
            dataset,
            descriptor_columns=["exact_match"],
            suite_id=suite["id"],
            descriptor_types={"exact_match": "ExactMatch"},
            output_column="answer",
        )

        validation = validate_result_set(result_set)
        assert validation.valid, validation.errors
        assert [r["test_case_id"] for r in result_set["results"]] == ["fr", "jp", "de"]
        assert [r["passed"] for r in result_set["results"]] == [True, True, False]
        assert result_set["results"][2]["actual_output"] == "Munich"
        gr0 = result_set["results"][0]["grader_results"][0]
        assert gr0["type"] == "exact_match"
        assert gr0["score"] == 1.0

    def test_real_text_length_and_contains_descriptors(self, df):
        suite = to_openeval(df, input_columns=["question"], ids=["fr", "jp", "de"])
        result_df = from_openeval(suite)

        dataset = Dataset.from_pandas(
            result_df,
            data_definition=DataDefinition(),
            descriptors=[
                TextLength("answer", alias="answer_length"),
                Contains("answer", items=["Paris"], alias="contains_paris"),
            ],
        )
        result_set = evaluation_result_to_openeval(
            dataset,
            descriptor_columns=["answer_length", "contains_paris"],
            suite_id=suite["id"],
        )
        assert validate_result_set(result_set).valid

        # "Paris"/"Tokyo"/"Munich" -> lengths 5/5/6, all raw > 1.0 -> clamped
        lengths = [
            r["grader_results"][0] for r in result_set["results"]
        ]
        assert [g["score"] for g in lengths] == [1.0, 1.0, 1.0]
        assert [g["metadata"]["evidently"]["raw_score"] for g in lengths] == [5.0, 5.0, 6.0]

        contains = [r["grader_results"][1] for r in result_set["results"]]
        assert [g["score"] for g in contains] == [1.0, 0.0, 0.0]
        assert [g["passed"] for g in contains] == [True, False, False]
        assert [r["passed"] for r in result_set["results"]] == [True, False, False]

    def test_descriptor_columns_required(self, df):
        with pytest.raises(ValueError, match="descriptor_columns is empty"):
            evaluation_result_to_openeval(df, descriptor_columns=[], suite_id="s")

    def test_missing_descriptor_column_raises(self, df):
        with pytest.raises(ValueError, match="not present"):
            evaluation_result_to_openeval(df, descriptor_columns=["nonexistent"], suite_id="s")

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="evaluated data is empty"):
            evaluation_result_to_openeval(
                pd.DataFrame({"exact_match": []}), descriptor_columns=["exact_match"], suite_id="s"
            )

    def test_bool_descriptor_score_and_passed(self):
        result_df = pd.DataFrame({"id": ["a", "b"], "exact_match": [True, False]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match"], suite_id="s"
        )
        grs = [r["grader_results"][0] for r in result_set["results"]]
        assert [g["score"] for g in grs] == [1.0, 0.0]
        assert [g["passed"] for g in grs] == [True, False]
        assert "reason" not in grs[0]
        assert "metadata" not in grs[0]

    def test_numeric_descriptor_clamping_preserves_raw_score(self):
        result_df = pd.DataFrame({"id": ["a", "b", "c"], "metric": [1.5, -0.2, 0.6]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["metric"], suite_id="s"
        )
        grs = [r["grader_results"][0] for r in result_set["results"]]
        assert [g["score"] for g in grs] == [1.0, 0.0, 0.6]
        assert grs[0]["metadata"]["evidently"]["raw_score"] == 1.5
        assert grs[1]["metadata"]["evidently"]["raw_score"] == -0.2
        assert "metadata" not in grs[2]
        assert validate_result_set(result_set).valid

    def test_non_numeric_descriptor_gets_null_score_and_reason(self):
        result_df = pd.DataFrame({"id": ["a", "b"], "label": ["POSITIVE", "NEGATIVE"]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["label"], suite_id="s"
        )
        grs = [r["grader_results"][0] for r in result_set["results"]]
        assert grs[0]["score"] is None
        assert grs[0]["reason"] == "POSITIVE"
        assert grs[0]["passed"] is False  # no pass_values given -> default False
        assert validate_result_set(result_set).valid

    def test_non_numeric_descriptor_pass_values_allowlist(self):
        result_df = pd.DataFrame({"id": ["a", "b"], "label": ["POSITIVE", "NEGATIVE"]})
        result_set = evaluation_result_to_openeval(
            result_df,
            descriptor_columns=["label"],
            suite_id="s",
            pass_values={"label": {"POSITIVE"}},
        )
        grs = [r["grader_results"][0] for r in result_set["results"]]
        assert [g["passed"] for g in grs] == [True, False]

    def test_null_descriptor_value_gets_null_score_and_fails(self):
        result_df = pd.DataFrame({"id": ["a"], "metric": [None]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["metric"], suite_id="s"
        )
        gr = result_set["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert gr["passed"] is False
        assert validate_result_set(result_set).valid

    def test_multiple_descriptors_and_row_level_passed(self):
        result_df = pd.DataFrame(
            {"id": ["a", "b", "c"], "exact_match": [True, True, False], "length_ok": [0.9, 0.9, 0.9]}
        )
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match", "length_ok"], suite_id="s"
        )
        assert [r["passed"] for r in result_set["results"]] == [True, True, False]
        assert len(result_set["results"][0]["grader_results"]) == 2

    def test_id_column_used_when_present(self):
        result_df = pd.DataFrame({"id": ["fr", "jp"], "exact_match": [True, True]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match"], suite_id="s"
        )
        assert [r["test_case_id"] for r in result_set["results"]] == ["fr", "jp"]

    def test_explicit_id_column_overrides_default(self):
        result_df = pd.DataFrame({"question": ["q1", "q2"], "exact_match": [True, True]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match"], suite_id="s", id_column="question"
        )
        assert result_set["results"][0]["test_case_id"] == "q1"

    def test_default_generated_ids_when_no_id_column(self):
        result_df = pd.DataFrame({"exact_match": [True, True]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match"], suite_id="s"
        )
        assert [r["test_case_id"] for r in result_set["results"]] == [
            "evidently_tc_0",
            "evidently_tc_1",
        ]

    def test_output_column_auto_detection(self):
        result_df = pd.DataFrame({"answer": ["Paris"], "exact_match": [True]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match"], suite_id="s"
        )
        assert result_set["results"][0]["actual_output"] == "Paris"

    def test_no_output_column_omits_actual_output(self):
        result_df = pd.DataFrame({"exact_match": [True]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match"], suite_id="s"
        )
        assert "actual_output" not in result_set["results"][0]

    def test_explicit_run_id_and_timestamps(self):
        result_df = pd.DataFrame({"exact_match": [True]})
        result_set = evaluation_result_to_openeval(
            result_df,
            descriptor_columns=["exact_match"],
            suite_id="s",
            run_id="fixed_run_id",
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:05:00Z",
        )
        assert result_set["run_id"] == "fixed_run_id"
        assert result_set["started_at"] == "2026-01-01T00:00:00Z"
        assert result_set["completed_at"] == "2026-01-01T00:05:00Z"

    def test_default_run_id_and_started_at_generated(self):
        result_df = pd.DataFrame({"exact_match": [True]})
        result_set = evaluation_result_to_openeval(
            result_df, descriptor_columns=["exact_match"], suite_id="s"
        )
        assert result_set["run_id"].startswith("evidently_run_")
        assert result_set["started_at"]
        assert "completed_at" not in result_set

    def test_accepts_object_with_as_dataframe_method(self):
        class FakeDataset:
            def as_dataframe(self):
                return pd.DataFrame({"id": ["a"], "exact_match": [True]})

        result_set = evaluation_result_to_openeval(
            FakeDataset(), descriptor_columns=["exact_match"], suite_id="s"
        )
        assert validate_result_set(result_set).valid
        assert len(result_set["results"]) == 1


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestFullLoop:
    def test_full_loop_id_preservation_and_validation(self, df):
        suite = to_openeval(
            df,
            input_columns=["question"],
            expected_column="expected",
            graders=["exact_match"],
            descriptor_types={"exact_match": "ExactMatch"},
            ids=["fr", "jp", "de"],
            suite_id="capitals_suite",
        )
        assert validate_suite(suite).valid

        result_df = from_openeval(suite)
        dataset = Dataset.from_pandas(
            result_df,
            data_definition=DataDefinition(),
            descriptors=[ExactMatch(columns=["expected", "answer"], alias="exact_match")],
        )
        result_set = evaluation_result_to_openeval(
            dataset, descriptor_columns=["exact_match"], suite_id=suite["id"], output_column="answer"
        )
        assert validate_result_set(result_set).valid

        suite_ids = [tc["id"] for tc in suite["test_cases"]]
        result_ids = [r["test_case_id"] for r in result_set["results"]]
        assert suite_ids == result_ids == ["fr", "jp", "de"]
