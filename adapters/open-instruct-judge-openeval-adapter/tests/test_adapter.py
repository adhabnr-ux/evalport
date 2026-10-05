import pytest

from openeval.validate import validate_result_set
from openeval.types import OPENEVAL_VERSION

from open_instruct_judge_openeval_adapter import to_openeval, from_openeval, KNOWN_JUDGE_TYPES


# ---------------------------------------------------------------------------
# 1. Valid JSON judge response (the common path)
# ---------------------------------------------------------------------------
def test_valid_json_quality_response_normalizes_and_validates():
    examples = [
        {
            "id": "ex1",
            "judge_type": "quality",
            "input": "What is the capital of France?",
            "output": "Paris.",
            "raw_judge_response": '{"REASONING": "Correct and concise.", "SCORE": "8"}',
            "judge_model": "gpt-4o",
        }
    ]
    rs = to_openeval(examples, suite_id="oi_judge_eval", run_id="run1")

    assert rs["version"] == OPENEVAL_VERSION
    result = rs["results"][0]
    gr = result["grader_results"][0]
    assert gr["score"] == pytest.approx(0.8)  # 8 / 10, divided exactly once
    assert gr["reason"] == "Correct and concise."
    assert gr["metadata"]["parse_status"] == "json"
    assert gr["metadata"]["extractor"] == "extract_score_with_fallback_max_10"
    assert result["passed"] is True  # 0.8 >= default 0.5 threshold

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_valid_json_strips_markdown_fence():
    # No embedded newlines inside the fence: a clean json.loads() path.
    examples = [
        {
            "id": "ex_fenced",
            "judge_type": "quality_rubric",
            "input": "q",
            "output": "a",
            "label": "- must mention x",
            "raw_judge_response": '```json{"REASONING": "Meets all criteria.", "SCORE": "1"}```',
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["metadata"]["parse_status"] == "json"
    assert validate_result_set(rs).valid


def test_fenced_multiline_json_falls_back_to_regex():
    # Faithfully mirrors a real upstream quirk in extract_json_score_with_fallback():
    # the function escapes actual newlines to literal "\n" *after* stripping the
    # ```json fence, so a fence on its own line leaves a literal leading "\n"
    # immediately before "{", which is not valid at the start of a JSON document.
    # json.loads() genuinely fails here (upstream too) and falls through to the
    # regex fallback -- which still recovers the real score correctly.
    examples = [
        {
            "id": "ex_fenced_multiline",
            "judge_type": "quality_rubric",
            "input": "q",
            "output": "a",
            "raw_judge_response": '```json\n{"REASONING": "Meets all criteria.", "SCORE": "1"}\n```',
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["metadata"]["parse_status"] == "regex_fallback"
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# 2. Fallback (regex) parsing -- JSON itself fails but SCORE is recoverable
# ---------------------------------------------------------------------------
def test_fallback_regex_parsing_when_json_is_broken():
    # Unbalanced braces / stray text around a recoverable "SCORE" key --
    # json.loads() will raise, but the regex fallback still finds it.
    raw = 'Some preamble text {"REASONING": "ok" "SCORE": "6"'
    examples = [
        {
            "id": "ex2",
            "judge_type": "quality_ref",
            "input": "in",
            "output": "out",
            "label": "ref",
            "raw_judge_response": raw,
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    gr = rs["results"][0]["grader_results"][0]
    assert gr["metadata"]["parse_status"] == "regex_fallback"
    assert gr["score"] == pytest.approx(0.6)  # 6 / 10
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# 3. A legitimate zero -- must NOT be confused with a parse failure
# ---------------------------------------------------------------------------
def test_legitimate_zero_is_not_a_parse_failure():
    examples = [
        {
            "id": "ex3",
            "judge_type": "quality",
            "input": "in",
            "output": "irrelevant nonsense",
            "raw_judge_response": '{"REASONING": "Completely off-topic.", "SCORE": "0"}',
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    gr = rs["results"][0]["grader_results"][0]
    assert gr["metadata"]["parse_status"] == "json"  # genuinely parsed, not "failed"
    assert gr["score"] == 0.0
    assert gr["passed"] is False
    assert rs["results"][0]["passed"] is False
    assert validate_result_set(rs).valid


def test_web_instruct_verifier_binary_scores():
    examples = [
        {"id": "yes1", "judge_type": "web_instruct_general_verifier", "input": "2+2?", "output": "4",
         "raw_judge_response": "Reasoning... Final Decision: Yes"},
        {"id": "no1", "judge_type": "web_instruct_general_verifier", "input": "2+2?", "output": "5",
         "raw_judge_response": "Reasoning... Final Decision: No"},
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    scores = {r["test_case_id"]: r["grader_results"][0]["score"] for r in rs["results"]}
    assert scores == {"yes1": 1.0, "no1": 0.0}
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# 4. Malformed / unparseable output -- score: null, not a silent 0.0
# ---------------------------------------------------------------------------
def test_malformed_output_is_unverified_not_zero():
    examples = [
        {
            "id": "ex4",
            "judge_type": "safety",
            "input": "in",
            "output": "out",
            "raw_judge_response": "the model just rambled with no SCORE field at all",
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    result = rs["results"][0]
    gr = result["grader_results"][0]

    assert gr["metadata"]["parse_status"] == "failed"
    assert gr["score"] is None  # NOT 0.0 -- "not verified", per SPEC.md Rule 6
    assert gr["passed"] is False
    assert result["passed"] is False
    # Raw response is preserved so the failure can still be inspected.
    assert gr["metadata"]["raw_judge_response"] == examples[0]["raw_judge_response"]

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


def test_malformed_web_verifier_output_is_also_unverified():
    examples = [
        {
            "id": "ex4b",
            "judge_type": "web_instruct_general_verifier",
            "input": "in",
            "output": "out",
            "raw_judge_response": "the model never committed to a final decision",
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    gr = rs["results"][0]["grader_results"][0]
    assert gr["metadata"]["parse_status"] == "failed"
    assert gr["score"] is None
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# 5. A batch spanning more than one judge type, validated as a whole
# ---------------------------------------------------------------------------
def test_batch_with_multiple_judge_types_validates_as_one_result_set():
    examples = [
        {"id": "q1", "judge_type": "quality", "input": "i1", "output": "o1",
         "raw_judge_response": '{"REASONING": "fine", "SCORE": "7"}'},
        {"id": "qr1", "judge_type": "quality_rubric", "input": "i2", "output": "o2", "label": "rubric",
         "raw_judge_response": '{"REASONING": "meets bar", "SCORE": "1"}'},
        {"id": "f1", "judge_type": "factuality", "input": "i3", "output": "o3", "label": "ref3",
         "raw_judge_response": '{"REASONING": "matches reference", "SCORE": "1"}'},
        {"id": "wv1", "judge_type": "web_instruct_general_verifier", "input": "i4", "output": "o4",
         "raw_judge_response": "Final Decision: Yes"},
        {"id": "fail1", "judge_type": "refusal", "input": "i5", "output": "o5",
         "raw_judge_response": "not parseable at all"},
    ]
    rs = to_openeval(examples, suite_id="oi_mixed_batch", run_id="run_mixed")

    assert len(rs["results"]) == 5
    assert {r["test_case_id"] for r in rs["results"]} == {"q1", "qr1", "f1", "wv1", "fail1"}
    fail_result = next(r for r in rs["results"] if r["test_case_id"] == "fail1")
    assert fail_result["grader_results"][0]["score"] is None

    validation = validate_result_set(rs)
    assert validation.valid, validation.errors


# ---------------------------------------------------------------------------
# Precomputed scores: already normalized by the real EXTRACTOR_MAP, must
# never be divided again.
# ---------------------------------------------------------------------------
def test_precomputed_score_is_not_renormalized():
    examples = [
        {
            "id": "ex5",
            "judge_type": "quality",  # a /10 judge type
            "input": "in",
            "output": "out",
            "score": 0.07,  # already run through extract_score_with_fallback_max_10
            "reasoning": "already extracted upstream",
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] == pytest.approx(0.07)  # NOT 0.007
    assert gr["metadata"]["parse_status"] == "precomputed"
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# Out-of-range scores get clipped (and flagged), never rejected
# ---------------------------------------------------------------------------
def test_out_of_range_score_is_clipped_and_flagged():
    examples = [
        {
            "id": "ex6",
            "judge_type": "factuality",
            "input": "in",
            "output": "out",
            "raw_judge_response": '{"REASONING": "weird scale", "SCORE": "5"}',
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    gr = rs["results"][0]["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["metadata"]["score_clipped"] is True
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------
def test_unknown_judge_type_raises_value_error():
    examples = [{"id": "x", "judge_type": "not_a_real_judge_type", "input": "i", "output": "o", "score": 1.0}]
    with pytest.raises(ValueError, match="unknown judge_type"):
        to_openeval(examples, suite_id="s", run_id="r")


def test_missing_score_and_raw_response_raises_value_error():
    examples = [{"id": "x", "judge_type": "quality", "input": "i", "output": "o"}]
    with pytest.raises(ValueError, match="need either a precomputed"):
        to_openeval(examples, suite_id="s", run_id="r")


def test_empty_batch_raises_value_error():
    with pytest.raises(ValueError, match="non-empty"):
        to_openeval([], suite_id="s", run_id="r")


def test_known_judge_types_matches_documented_set():
    assert KNOWN_JUDGE_TYPES == {
        "quality",
        "quality_rubric",
        "quality_ref",
        "safety",
        "factuality",
        "creative_writing",
        "refusal",
        "web_instruct_general_verifier",
    }


# ---------------------------------------------------------------------------
# Attribute-style objects (not just dicts) work the same way
# ---------------------------------------------------------------------------
class FakeJudgedExample:
    def __init__(self, id, judge_type, input, output, raw_judge_response, label=None):
        self.id = id
        self.judge_type = judge_type
        self.input = input
        self.output = output
        self.raw_judge_response = raw_judge_response
        self.label = label


def test_to_openeval_from_attribute_objects():
    examples = [
        FakeJudgedExample(
            "obj1", "quality", "in", "out", '{"REASONING": "good", "SCORE": "9"}'
        )
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    assert rs["results"][0]["grader_results"][0]["score"] == pytest.approx(0.9)
    assert validate_result_set(rs).valid


# ---------------------------------------------------------------------------
# Stable IDs, round-tripping, and index-based fallback
# ---------------------------------------------------------------------------
def test_fallback_id_when_none_provided():
    examples = [{"judge_type": "quality", "input": "i", "output": "o", "score": 0.5}]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    assert rs["results"][0]["test_case_id"] == "tc_0"


def test_from_openeval_round_trip():
    examples = [
        {
            "id": "rt1",
            "judge_type": "quality_ref",
            "input": "question",
            "output": "answer",
            "label": "reference answer",
            "raw_judge_response": '{"REASONING": "close enough", "SCORE": "10"}',
            "judge_model": "gpt-4o",
        }
    ]
    rs = to_openeval(examples, suite_id="s", run_id="r")
    assert validate_result_set(rs).valid

    round_tripped = from_openeval(rs)
    assert len(round_tripped) == 1
    rt = round_tripped[0]
    assert rt["id"] == "rt1"
    assert rt["judge_type"] == "quality_ref"
    assert rt["score"] == pytest.approx(1.0)
    assert rt["input"] == "question"
    assert rt["output"] == "answer"
    assert rt["label"] == "reference answer"
    assert rt["judge_model"] == "gpt-4o"
    assert rt["parse_status"] == "json"
    assert rt["passed"] is True


def test_custom_pass_threshold():
    examples = [
        {"id": "a", "judge_type": "quality", "input": "i", "output": "o", "score": 0.6},
        {"id": "b", "judge_type": "quality", "input": "i", "output": "o", "score": 0.6},
    ]
    strict = to_openeval(examples, suite_id="s", run_id="r", pass_threshold=0.9)
    assert all(r["passed"] is False for r in strict["results"])
    lenient = to_openeval(examples, suite_id="s", run_id="r", pass_threshold=0.5)
    assert all(r["passed"] is True for r in lenient["results"])
