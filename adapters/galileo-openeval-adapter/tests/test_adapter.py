"""Tests for galileo_openeval_adapter.

Runs against the real, installed `galileo` package (Dataset, Trace, LlmSpan,
Message, LocalMetric -- no mocks, no reinvented stand-ins) and the real
`openeval.validate` validators. `galileo` is a client for a hosted SaaS
product, so anything that would require a live, authenticated API call
(`Dataset.create()`/`.get_content()`, `LogStream`/`Experiment` execution,
built-in `GalileoMetric` scoring) is deliberately out of scope here -- these
tests exercise the fully-offline surface: local `Dataset(content=...)`
construction, local `Trace`/`Span` construction, and real
`LocalMetric.scorer_fn` calls.
"""
import pytest

from galileo import Dataset, Trace, LlmSpan, Message, MessageRole, LocalMetric
from galileo.metric import GalileoMetric

from openeval.validate import validate_suite, validate_result_set

from galileo_openeval_adapter import to_openeval, from_openeval, spans_to_openeval
from galileo_openeval_adapter import _extract_text


# ---------------------------------------------------------------------------
# Fixtures / builders using the real galileo classes
# ---------------------------------------------------------------------------

def _basic_content(**overrides):
    row = {"input": "What is the capital of France?", "output": "Paris"}
    row.update(overrides)
    return [row]


def _full_row():
    return {
        "input": "What is the capital of France?",
        "output": "Paris",
        "difficulty": "easy",
        "category": "geography",
        "tags": ["factual", "short-answer"],
    }


def _length_scorer(trace_or_span):
    # LlmSpan coerces a plain-string `output` into a real Message(content=...,
    # role=assistant) at construction time -- confirmed empirically, not
    # assumed -- while Trace leaves a plain-string `output` untouched. Reuse
    # the adapter's own `_extract_text` (already proven against both shapes)
    # rather than hand-rolling a second, weaker extraction here.
    text = _extract_text(getattr(trace_or_span, "output", None)) or ""
    return min(len(text) / 20.0, 1.0)


def _explainable_scorer(trace_or_span):
    text = _extract_text(getattr(trace_or_span, "output", None)) or ""
    matched = [k for k in ("paris", "france") if k in text.lower()]
    return len(matched) / 2.0, {"matched": matched}


def _label_scorer(trace_or_span):
    text = _extract_text(getattr(trace_or_span, "output", None)) or ""
    return "correct" if "paris" in text.lower() else "incorrect"


def _local_metric(name, fn):
    return LocalMetric(name=name, scorer_fn=fn)


# ---------------------------------------------------------------------------
# to_openeval()
# ---------------------------------------------------------------------------

class TestToOpeneval:
    def test_basic_row_produces_valid_suite(self):
        suite = to_openeval(_basic_content(), suite_id="geo_quiz")
        result = validate_suite(suite)
        assert result.valid, result.errors
        tc = suite["test_cases"][0]
        assert tc["input"] == "What is the capital of France?"
        assert tc["expected_output"] == "Paris"
        assert tc["graders"] == ["gr_galileo_metrics"]

    def test_real_dataset_content_accepted(self):
        # Exactly what galileo.Dataset(content=...) itself accepts, per its
        # real constructor -- confirms this adapter reads the same shape.
        dataset = Dataset(name="geo-quiz", content=_basic_content())
        suite = to_openeval(dataset.content, suite_id="geo_quiz")
        assert validate_suite(suite).valid

    def test_missing_input_key_raises(self):
        with pytest.raises(ValueError, match="input"):
            to_openeval([{"output": "Paris"}], suite_id="s1")

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="input"):
            to_openeval([{"input": "", "output": "Paris"}], suite_id="s1")

    def test_custom_input_key_and_expected_output_key(self):
        rows = [{"question": "2+2?", "answer": "4"}]
        suite = to_openeval(rows, suite_id="s1", input_key="question", expected_output_key="answer")
        tc = suite["test_cases"][0]
        assert tc["input"] == "2+2?"
        assert tc["expected_output"] == "4"
        assert validate_suite(suite).valid

    def test_multiturn_list_input(self):
        rows = [{"input": ["Hi", "What's the capital of France?"], "output": "Paris"}]
        suite = to_openeval(rows, suite_id="s1")
        assert suite["test_cases"][0]["input"] == ["Hi", "What's the capital of France?"]
        assert validate_suite(suite).valid

    def test_multiturn_list_with_non_string_items_raises(self):
        rows = [{"input": ["Hi", {"role": "user", "content": "hey"}], "output": "x"}]
        with pytest.raises(ValueError, match="multi-turn"):
            to_openeval(rows, suite_id="s1")

    def test_non_string_non_list_input_raises(self):
        with pytest.raises(ValueError, match="str or list"):
            to_openeval([{"input": 42, "output": "x"}], suite_id="s1")

    def test_explicit_ids(self):
        suite = to_openeval(_basic_content() * 2, suite_id="s1", ids=["a", "b"])
        assert [tc["id"] for tc in suite["test_cases"]] == ["a", "b"]

    def test_row_id_field_used_when_present(self):
        rows = [{"id": "row-99", "input": "hi", "output": "hello"}]
        suite = to_openeval(rows, suite_id="s1")
        assert suite["test_cases"][0]["id"] == "row-99"

    def test_row_row_id_field_used_when_present(self):
        rows = [{"row_id": "r-5", "input": "hi", "output": "hello"}]
        suite = to_openeval(rows, suite_id="s1")
        assert suite["test_cases"][0]["id"] == "r-5"

    def test_default_id_is_positional(self):
        suite = to_openeval(_basic_content() * 3, suite_id="s1")
        assert [tc["id"] for tc in suite["test_cases"]] == ["tc_0", "tc_1", "tc_2"]

    def test_extra_columns_preserved_in_metadata(self):
        suite = to_openeval([_full_row()], suite_id="s1")
        row = suite["test_cases"][0]["metadata"]["galileo"]["row"]
        assert row == _full_row()

    def test_missing_expected_output_omitted(self):
        suite = to_openeval([{"input": "hi"}], suite_id="s1")
        assert "expected_output" not in suite["test_cases"][0]
        assert validate_suite(suite).valid

    def test_non_string_expected_output_coerced_to_str(self):
        rows = [{"input": "how many?", "output": 4}]
        suite = to_openeval(rows, suite_id="s1")
        assert suite["test_cases"][0]["expected_output"] == "4"

    def test_custom_grader_id_and_handler(self):
        suite = to_openeval(
            _basic_content(), suite_id="s1", grader_id="gr_custom", grader_handler="galileo:custom"
        )
        assert suite["test_cases"][0]["graders"] == ["gr_custom"]
        assert suite["graders"][0]["id"] == "gr_custom"
        assert suite["graders"][0]["params"]["handler"] == "galileo:custom"

    def test_name_param(self):
        suite = to_openeval(_basic_content(), suite_id="s1", name="My Geo Suite")
        assert suite["name"] == "My Geo Suite"

    def test_multiple_rows_all_converted(self):
        rows = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
            {"input": "Q3"},
        ]
        suite = to_openeval(rows, suite_id="s1")
        assert len(suite["test_cases"]) == 3
        assert "expected_output" not in suite["test_cases"][2]
        assert validate_suite(suite).valid


# ---------------------------------------------------------------------------
# from_openeval()
# ---------------------------------------------------------------------------

class TestFromOpeneval:
    def test_round_trip_restores_exact_original_row(self):
        original = [_full_row()]
        suite = to_openeval(original, suite_id="s1")
        restored = from_openeval(suite)
        assert restored == original

    def test_round_trip_works_with_real_dataset_constructor(self):
        original = _basic_content()
        suite = to_openeval(original, suite_id="s1")
        restored = from_openeval(suite)
        # Confirms the restored rows are actually usable by galileo.Dataset.
        dataset = Dataset(name="restored", content=restored)
        assert dataset.content == original

    def test_hand_authored_suite_without_galileo_metadata(self):
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "test_cases": [
                {"id": "tc1", "input": "What is 2+2?", "expected_output": "4", "graders": ["g1"]},
            ],
        }
        rows = from_openeval(suite)
        assert rows == [{"input": "What is 2+2?", "output": "4"}]

    def test_hand_authored_suite_no_expected_output(self):
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "test_cases": [{"id": "tc1", "input": "hi", "graders": ["g1"]}],
        }
        rows = from_openeval(suite)
        assert rows == [{"input": "hi"}]

    def test_multiturn_input_passed_through_as_list(self):
        suite = {
            "version": "1.0.0",
            "id": "s1",
            "test_cases": [{"id": "tc1", "input": ["hi", "how are you?"], "graders": ["g1"]}],
        }
        rows = from_openeval(suite)
        assert rows == [{"input": ["hi", "how are you?"]}]

    def test_empty_suite(self):
        suite = {"version": "1.0.0", "id": "s1", "test_cases": []}
        assert from_openeval(suite) == []


# ---------------------------------------------------------------------------
# spans_to_openeval()
# ---------------------------------------------------------------------------

class TestSpansToOpeneval:
    def test_basic_numeric_score(self):
        span = LlmSpan(input="What is the capital of France?", output="Paris is the capital of France.")
        metric = _local_metric("response_length", _length_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        assert validate_result_set(rs).valid
        gr = rs["results"][0]["grader_results"][0]
        assert gr["grader_id"] == "response_length"
        assert gr["type"] == "custom"
        assert 0.0 <= gr["score"] <= 1.0

    def test_actual_output_extracted_from_plain_string(self):
        span = LlmSpan(input="hi", output="hello there")
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        assert rs["results"][0]["actual_output"] == "hello there"

    def test_actual_output_extracted_from_message_dict(self):
        span = LlmSpan(
            input=[{"role": "user", "content": "hi"}],
            output={"role": "assistant", "content": "hello there"},
        )
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        assert rs["results"][0]["actual_output"] == "hello there"

    def test_actual_output_extracted_from_real_message_object(self):
        # LlmSpan.output coerces a plain str to Message(...) automatically
        # (see _length_scorer's comment above), but a caller can also pass a
        # real galileo.Message instance directly, e.g. from their own logging
        # code -- confirm that shape is handled too, not just the coerced one.
        span = LlmSpan(
            input=Message(content="hi", role=MessageRole.user),
            output=Message(content="hello there", role=MessageRole.assistant),
        )
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        assert rs["results"][0]["actual_output"] == "hello there"

    def test_extract_text_handles_message_list_directly(self):
        # Neither LlmSpan.output nor Trace.output actually accepts a plain
        # list of chat messages (confirmed empirically: LlmSpan rejects it
        # outright -- "output must be a Message, a string, or a dict" --
        # and Trace's `output` list variant is a different, multimodal
        # content-part shape). _extract_text's list-of-messages branch exists
        # for callers who pass span-like objects with a genuinely list-typed
        # output (e.g. a different Span subtype, or a plain dict/namedtuple
        # test double) -- exercised directly here rather than forced through
        # LlmSpan/Trace construction, which the real SDK itself won't allow.
        flattened = _extract_text([
            Message(content="Sure, I can help.", role=MessageRole.assistant),
            {"role": "assistant", "content": "The capital is Paris."},
        ])
        assert flattened == "assistant: Sure, I can help.\nassistant: The capital is Paris."

    def test_trace_object_also_works(self):
        trace = Trace(input="hi", output="hello there")
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval([trace], metrics=[metric], suite_id="s1", run_id="run-1")
        assert validate_result_set(rs).valid
        assert rs["results"][0]["actual_output"] == "hello there"

    def test_tuple_return_preserves_scorer_metadata(self):
        span = LlmSpan(input="capital of France?", output="Paris is the capital of France.")
        metric = _local_metric("keyword_coverage", _explainable_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] == 1.0
        assert gr["metadata"]["scorer_metadata"]["matched"] == ["paris", "france"]

    def test_non_numeric_string_score_becomes_null_with_raw_value(self):
        span = LlmSpan(input="capital of France?", output="Paris is the capital of France.")
        metric = _local_metric("label", _label_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        gr = rs["results"][0]["grader_results"][0]
        assert gr["score"] is None
        assert gr["metadata"]["raw_value"] == "correct"
        assert gr["passed"] is True  # truthy non-empty string
        assert validate_result_set(rs).valid

    def test_multiple_metrics_per_span(self):
        span = LlmSpan(input="capital of France?", output="Paris is the capital of France.")
        m1 = _local_metric("length", _length_scorer)
        m2 = _local_metric("keywords", _explainable_scorer)
        rs = spans_to_openeval([span], metrics=[m1, m2], suite_id="s1", run_id="run-1")
        grader_ids = {g["grader_id"] for g in rs["results"][0]["grader_results"]}
        assert grader_ids == {"length", "keywords"}

    def test_score_clamped_above_one(self):
        span = LlmSpan(input="x", output="y")
        metric = _local_metric("over", lambda s: 5.0)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        assert rs["results"][0]["grader_results"][0]["score"] == 1.0

    def test_score_clamped_below_zero(self):
        span = LlmSpan(input="x", output="y")
        metric = _local_metric("under", lambda s: -5.0)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        assert rs["results"][0]["grader_results"][0]["score"] == 0.0

    def test_pass_threshold_respected(self):
        span = LlmSpan(input="x", output="y")
        metric = _local_metric("half", lambda s: 0.5)
        rs_low = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1", pass_threshold=0.4)
        rs_high = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1", pass_threshold=0.6)
        assert rs_low["results"][0]["grader_results"][0]["passed"] is True
        assert rs_high["results"][0]["grader_results"][0]["passed"] is False

    def test_empty_metrics_raises(self):
        span = LlmSpan(input="x", output="y")
        with pytest.raises(ValueError, match="LocalMetric"):
            spans_to_openeval([span], metrics=[], suite_id="s1", run_id="run-1")

    def test_non_local_metric_raises(self):
        span = LlmSpan(input="x", output="y")
        built_in = GalileoMetric.metrics.correctness if hasattr(GalileoMetric, "metrics") else None
        # GalileoMetric.metrics.correctness is itself a GalileoMetric instance
        # with no local scorer_fn -- the real object, not a stub.
        from galileo import Metric
        correctness = Metric.metrics.correctness
        with pytest.raises(ValueError, match="scorer_fn"):
            spans_to_openeval([span], metrics=[correctness], suite_id="s1", run_id="run-1")

    def test_empty_spans_raises(self):
        metric = _local_metric("len", _length_scorer)
        with pytest.raises(ValueError, match="spans"):
            spans_to_openeval([], metrics=[metric], suite_id="s1", run_id="run-1")

    def test_ids_correlation(self):
        spans = [LlmSpan(input="a", output="1"), LlmSpan(input="b", output="2")]
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval(spans, metrics=[metric], suite_id="s1", run_id="run-1", ids=["x1", "x2"])
        assert [r["test_case_id"] for r in rs["results"]] == ["x1", "x2"]

    def test_default_ids_positional(self):
        spans = [LlmSpan(input="a", output="1"), LlmSpan(input="b", output="2")]
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval(spans, metrics=[metric], suite_id="s1", run_id="run-1")
        assert [r["test_case_id"] for r in rs["results"]] == ["tc_0", "tc_1"]

    def test_user_and_dataset_metadata_preserved(self):
        span = LlmSpan(input="a", output="1")
        span.user_metadata = {"env": "staging"}
        span.dataset_metadata = {"row_index": "3"}
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        meta = rs["results"][0]["metadata"]["galileo"]
        assert meta["user_metadata"] == {"env": "staging"}
        assert meta["dataset_metadata"] == {"row_index": "3"}

    def test_summary_counts_and_avg_score(self):
        spans = [LlmSpan(input="a", output="x" * 20), LlmSpan(input="b", output="")]
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval(spans, metrics=[metric], suite_id="s1", run_id="run-1", pass_threshold=0.5)
        summary = rs["summary"]
        assert summary["total"] == 2
        assert summary["passed"] == 1
        assert summary["failed"] == 1
        assert summary["pass_rate"] == 0.5
        assert summary["avg_score"] == pytest.approx((1.0 + 0.0) / 2)

    def test_explicit_started_at_and_completed_at(self):
        span = LlmSpan(input="a", output="1")
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval(
            [span], metrics=[metric], suite_id="s1", run_id="run-1",
            started_at="2026-08-22T00:00:00Z", completed_at="2026-08-22T00:01:00Z",
        )
        assert rs["started_at"] == "2026-08-22T00:00:00Z"
        assert rs["completed_at"] == "2026-08-22T00:01:00Z"

    def test_default_started_at_is_generated(self):
        span = LlmSpan(input="a", output="1")
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval([span], metrics=[metric], suite_id="s1", run_id="run-1")
        assert rs["started_at"]  # non-empty, real ISO timestamp
        assert rs["completed_at"] == rs["started_at"]

    def test_runner_name_and_version(self):
        span = LlmSpan(input="a", output="1")
        metric = _local_metric("len", _length_scorer)
        rs = spans_to_openeval(
            [span], metrics=[metric], suite_id="s1", run_id="run-1",
            runner_name="my-runner", runner_version="9.9.9",
        )
        assert rs["runner"] == {"name": "my-runner", "version": "9.9.9"}


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_suite_to_results_round_trip():
    """Suite -> Dataset -> simulated app run (real Span objects) ->
    real LocalMetric scoring -> ResultSet, validated against the real spec
    at both ends."""
    content = [
        {"input": "What is the capital of France?", "output": "Paris"},
        {"input": "What is the capital of Japan?", "output": "Tokyo"},
    ]
    suite = to_openeval(content, suite_id="geo_quiz", ids=["q1", "q2"])
    assert validate_suite(suite).valid

    rows = from_openeval(suite)
    dataset = Dataset(name="geo-quiz", content=rows)
    assert dataset.content == content

    # Simulate running the app against each row and logging a real Span.
    fake_app_outputs = {"q1": "Paris is the capital of France.", "q2": "Kyoto"}  # q2 deliberately wrong
    spans = [
        LlmSpan(input=row["input"], output=fake_app_outputs[qid])
        for row, qid in zip(rows, ["q1", "q2"])
    ]

    def _exact_match_scorer(trace_or_span):
        text = (_extract_text(trace_or_span.output) or "").lower()
        expected = {"q1": "paris", "q2": "tokyo"}
        # crude closure-free lookup by output content presence
        return 1.0 if any(exp in text for exp in expected.values()) else 0.0

    metric = _local_metric("contains_expected_city", _exact_match_scorer)
    result_set = spans_to_openeval(spans, metrics=[metric], suite_id="geo_quiz", run_id="run-1", ids=["q1", "q2"])
    assert validate_result_set(result_set).valid
    assert result_set["results"][0]["grader_results"][0]["score"] == 1.0  # "Paris" matched
    assert result_set["results"][1]["grader_results"][0]["score"] == 0.0  # "Kyoto" != "Tokyo"
    assert result_set["summary"]["total"] == 2
