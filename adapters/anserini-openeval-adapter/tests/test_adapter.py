import math

import pytest

from openeval.validate import validate_suite, validate_result_set

from anserini_openeval_adapter import (
    average_precision,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    parse_run_file,
    parse_qrels_file,
    parse_topics_file,
    to_openeval,
    from_openeval,
)

# Same tiny fixture used across the metric unit tests and the file-based
# to_openeval() tests below, hand-computed once here so every assertion
# traces back to the same worked example.
#
# qrels for q1: d1=1, d2=0, d3=1, d4=2   (judged-relevant: d1, d3, d4)
# run for q1, ranked:  d2 (rank1), d1 (rank2), d5 (rank3, unjudged),
#                       d3 (rank4), d4 (rank5)
Q1_QRELS = {"d1": 1, "d2": 0, "d3": 1, "d4": 2}
Q1_RANKED = ["d2", "d1", "d5", "d3", "d4"]


def test_average_precision_hand_computed():
    # Relevant hits at rank 2 (d1, P=1/2), rank 4 (d3, P=2/4), rank 5 (d4, P=3/5).
    # AP = (1/3) * (0.5 + 0.5 + 0.6) = 0.5333...
    ap = average_precision(Q1_RANKED, Q1_QRELS)
    assert ap == pytest.approx(8 / 15, rel=1e-9)


def test_average_precision_no_relevant_judged_is_zero():
    assert average_precision(["a", "b"], {"a": 0, "b": 0}) == 0.0
    assert average_precision(["a", "b"], {}) == 0.0


def test_ndcg_at_5_hand_computed():
    # DCG@5 uses gains [0, 1, 0, 1, 2] at ranks [1..5] (rank1=d2 rel0,
    # rank2=d1 rel1, rank3=d5 unjudged rel0, rank4=d3 rel1, rank5=d4 rel2):
    #   DCG@5 = 0/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5) + 2/log2(6)
    # IDCG@5 uses the ideal ordering of judged-relevant gains [2, 1, 1]
    # padded with 0s to length 5: [2, 1, 1, 0, 0]:
    #   IDCG@5 = 2/log2(2) + 1/log2(3) + 1/log2(4) + 0 + 0
    dcg = (0 / math.log2(2)) + (1 / math.log2(3)) + (0 / math.log2(4)) + (1 / math.log2(5)) + (2 / math.log2(6))
    idcg = (2 / math.log2(2)) + (1 / math.log2(3)) + (1 / math.log2(4))
    expected = dcg / idcg
    assert ndcg_at_k(Q1_RANKED, Q1_QRELS, 5) == pytest.approx(expected, rel=1e-9)
    assert 0.0 < expected < 1.0  # sanity: imperfect ranking scores strictly between 0 and 1


def test_ndcg_perfect_ranking_is_one():
    # Ideal ordering by relevance (d4=2, d1=1, d3=1, d2=0) scores nDCG == 1.0
    # against its own qrels, regardless of tie order among equal-relevance docs.
    assert ndcg_at_k(["d4", "d1", "d3", "d2"], Q1_QRELS, 4) == pytest.approx(1.0, rel=1e-9)


def test_ndcg_no_relevant_judged_is_zero():
    assert ndcg_at_k(["a", "b"], {"a": 0, "b": 0}, 10) == 0.0


def test_recall_at_k_hand_computed():
    # Relevant = {d1, d3, d4}.
    assert recall_at_k(Q1_RANKED, Q1_QRELS, 5) == pytest.approx(1.0, rel=1e-9)  # all 3 retrieved by rank 5
    assert recall_at_k(Q1_RANKED, Q1_QRELS, 3) == pytest.approx(1 / 3, rel=1e-9)  # only d1 in top 3
    assert recall_at_k(Q1_RANKED, Q1_QRELS, 1) == pytest.approx(0.0, rel=1e-9)  # d2 (rel 0) at rank 1


def test_recall_no_relevant_judged_is_zero():
    assert recall_at_k(["a", "b"], {"a": 0}, 10) == 0.0


def test_reciprocal_rank_hand_computed():
    # First relevant doc (d1) is at rank 2.
    assert reciprocal_rank(Q1_RANKED, Q1_QRELS) == pytest.approx(0.5, rel=1e-9)


def test_reciprocal_rank_no_relevant_retrieved():
    assert reciprocal_rank(["d2"], Q1_QRELS) == 0.0  # only d2 (rel 0) retrieved
    assert reciprocal_rank([], Q1_QRELS) == 0.0


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------


def test_parse_run_file_standard_trec_format(tmp_path):
    run_file = tmp_path / "run.trec"
    run_file.write_text(
        "q1 Q0 d2 1 0.90 myrun\n"
        "q1 Q0 d1 2 0.80 myrun\n"
        "q2 Q0 dA 1 0.50 myrun\n"
        "\n"  # blank line should be skipped
    )
    runs = parse_run_file(str(run_file))
    assert runs["q1"] == [("d2", 1, 0.90), ("d1", 2, 0.80)]
    assert runs["q2"] == [("dA", 1, 0.50)]


def test_parse_run_file_msmarco_format(tmp_path):
    run_file = tmp_path / "run.msmarco.tsv"
    run_file.write_text("q1\td2\t1\nq1\td1\t2\n")
    runs = parse_run_file(str(run_file))
    assert runs["q1"] == [("d2", 1, None), ("d1", 2, None)]


def test_parse_run_file_out_of_order_lines_get_sorted_by_rank(tmp_path):
    run_file = tmp_path / "run.trec"
    run_file.write_text("q1 Q0 d1 2 0.5 r\nq1 Q0 d2 1 0.9 r\n")
    runs = parse_run_file(str(run_file))
    assert [docid for docid, _rank, _score in runs["q1"]] == ["d2", "d1"]


def test_parse_run_file_duplicate_docid_keeps_first(tmp_path):
    run_file = tmp_path / "run.trec"
    run_file.write_text("q1 Q0 d1 1 0.9 r\nq1 Q0 d1 2 0.5 r\n")
    runs = parse_run_file(str(run_file))
    assert runs["q1"] == [("d1", 1, 0.9)]


def test_parse_run_file_malformed_line_raises(tmp_path):
    run_file = tmp_path / "run.bad"
    run_file.write_text("q1 d1\n")
    with pytest.raises(ValueError, match="unrecognized run-file line"):
        parse_run_file(str(run_file))


def test_parse_qrels_file(tmp_path):
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 1\nq1 0 d2 0\nq2 0 dA 2\n")
    qrels = parse_qrels_file(str(qrels_file))
    assert qrels == {"q1": {"d1": 1, "d2": 0}, "q2": {"dA": 2}}


def test_parse_qrels_file_malformed_line_raises(tmp_path):
    qrels_file = tmp_path / "qrels.bad"
    qrels_file.write_text("q1 0 d1\n")
    with pytest.raises(ValueError, match="expected 4 fields"):
        parse_qrels_file(str(qrels_file))


def test_parse_topics_file(tmp_path):
    topics_file = tmp_path / "topics.tsv"
    topics_file.write_text("q1\twhat is trec_eval\nq2\thow does anserini index text\n")
    topics = parse_topics_file(str(topics_file))
    assert topics == {"q1": "what is trec_eval", "q2": "how does anserini index text"}


def test_parse_topics_file_missing_tab_raises(tmp_path):
    topics_file = tmp_path / "topics.bad"
    topics_file.write_text("q1 no tab here\n")
    with pytest.raises(ValueError, match="no tab found"):
        parse_topics_file(str(topics_file))


# ---------------------------------------------------------------------------
# to_openeval / from_openeval, end to end, validated against the real
# EvalPort validators (not a mock)
# ---------------------------------------------------------------------------


@pytest.fixture
def run_qrels_topics(tmp_path):
    run_file = tmp_path / "run.trec"
    run_file.write_text(
        "q1 Q0 d2 1 0.90 anserini_bm25\n"
        "q1 Q0 d1 2 0.80 anserini_bm25\n"
        "q1 Q0 d5 3 0.70 anserini_bm25\n"
        "q1 Q0 d3 4 0.60 anserini_bm25\n"
        "q1 Q0 d4 5 0.50 anserini_bm25\n"
        "q2 Q0 dX 1 0.40 anserini_bm25\n"
    )
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 1\nq1 0 d2 0\nq1 0 d3 1\nq1 0 d4 2\nq2 0 dX 1\n")
    topics_file = tmp_path / "topics.tsv"
    topics_file.write_text("q1\tsample query one\nq2\tsample query two\n")
    return str(run_file), str(qrels_file), str(topics_file)


def test_to_openeval_produces_valid_suite_and_result_set(run_qrels_topics):
    run_file, qrels_file, topics_file = run_qrels_topics
    suite, result_set = to_openeval(
        run_file, qrels_file, topics_file, suite_id="test-suite", run_id="test-run-1"
    )

    suite_validation = validate_suite(suite)
    assert suite_validation.valid, suite_validation.errors
    rs_validation = validate_result_set(result_set)
    assert rs_validation.valid, rs_validation.errors

    assert suite["id"] == "test-suite"
    assert result_set["suite_id"] == "test-suite"
    assert result_set["run_id"] == "test-run-1"
    assert len(suite["test_cases"]) == 2
    assert len(result_set["results"]) == 2


def test_to_openeval_test_case_shape(run_qrels_topics):
    run_file, qrels_file, topics_file = run_qrels_topics
    suite, _result_set = to_openeval(run_file, qrels_file, topics_file, suite_id="s")
    tc1 = next(tc for tc in suite["test_cases"] if tc["id"] == "q1")
    assert tc1["input"] == "sample query one"
    assert tc1["retrieval_context"] == ["d2", "d1", "d5", "d3", "d4"]
    assert tc1["graders"] == ["gr_map", "gr_ndcg_cut_10", "gr_recall_100"]


def test_to_openeval_without_topics_file_falls_back_to_qid(run_qrels_topics):
    run_file, qrels_file, _topics_file = run_qrels_topics
    suite, _result_set = to_openeval(run_file, qrels_file, suite_id="s")
    tc1 = next(tc for tc in suite["test_cases"] if tc["id"] == "q1")
    assert tc1["input"] == "q1"


def test_to_openeval_scores_match_direct_metric_calls(run_qrels_topics):
    run_file, qrels_file, topics_file = run_qrels_topics
    suite, result_set = to_openeval(
        run_file, qrels_file, topics_file, metrics=("map", "ndcg_cut_5", "recall_100", "mrr"), suite_id="s"
    )
    q1_result = next(r for r in result_set["results"] if r["test_case_id"] == "q1")
    scores = {gr["grader_id"]: gr["score"] for gr in q1_result["grader_results"]}

    ranked = ["d2", "d1", "d5", "d3", "d4"]
    assert scores["gr_map"] == pytest.approx(average_precision(ranked, Q1_QRELS), rel=1e-9)
    assert scores["gr_ndcg_cut_5"] == pytest.approx(ndcg_at_k(ranked, Q1_QRELS, 5), rel=1e-9)
    assert scores["gr_recall_100"] == pytest.approx(recall_at_k(ranked, Q1_QRELS, 100), rel=1e-9)
    assert scores["gr_mrr"] == pytest.approx(reciprocal_rank(ranked, Q1_QRELS), rel=1e-9)


def test_to_openeval_custom_grader_handler_names_are_traceable(run_qrels_topics):
    run_file, qrels_file, topics_file = run_qrels_topics
    suite, _result_set = to_openeval(run_file, qrels_file, topics_file, metrics=("map",), suite_id="s")
    grader = suite["graders"][0]
    assert grader["type"] == "custom"
    assert grader["params"]["handler"] == "anserini_openeval_adapter:map"


def test_to_openeval_unknown_query_ids_between_run_and_qrels_raises(tmp_path):
    run_file = tmp_path / "run.trec"
    run_file.write_text("qZZZ Q0 d1 1 0.9 r\n")
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("qOTHER 0 d1 1\n")
    with pytest.raises(ValueError, match="no query id in the run file"):
        to_openeval(str(run_file), str(qrels_file), suite_id="s")


def test_to_openeval_empty_run_file_raises(tmp_path):
    run_file = tmp_path / "run.trec"
    run_file.write_text("")
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 1\n")
    with pytest.raises(ValueError, match="no queries"):
        to_openeval(str(run_file), str(qrels_file), suite_id="s")


def test_to_openeval_unknown_metric_name_raises(run_qrels_topics):
    run_file, qrels_file, topics_file = run_qrels_topics
    with pytest.raises(ValueError, match="unknown metric"):
        to_openeval(run_file, qrels_file, topics_file, metrics=("not_a_real_metric",), suite_id="s")


def test_from_openeval_reconstructs_run_lines(run_qrels_topics):
    run_file, qrels_file, topics_file = run_qrels_topics
    suite, _result_set = to_openeval(run_file, qrels_file, topics_file, suite_id="s")
    reconstructed = from_openeval(suite, runtag="myrun")
    lines = reconstructed.strip("\n").split("\n")
    # 5 docs for q1 + 1 doc for q2 = 6 lines total.
    assert len(lines) == 6
    assert lines[0] == "q1 Q0 d2 1 1.000000 myrun"
    assert lines[1] == "q1 Q0 d1 2 0.500000 myrun"
    for line in lines:
        fields = line.split()
        assert len(fields) == 6
        assert fields[1] == "Q0"
        assert fields[5] == "myrun"


def test_from_openeval_missing_retrieval_context_raises():
    suite = {
        "version": "1.0.0",
        "id": "s",
        "test_cases": [{"id": "q1", "input": "x", "graders": ["g1"]}],
    }
    with pytest.raises(ValueError, match="no retrieval_context"):
        from_openeval(suite)


def test_msmarco_format_round_trip_through_to_openeval(tmp_path):
    run_file = tmp_path / "run.msmarco.tsv"
    run_file.write_text("q1\td2\t1\nq1\td1\t2\nq1\td3\t3\n")
    qrels_file = tmp_path / "qrels.txt"
    qrels_file.write_text("q1 0 d1 1\nq1 0 d3 1\n")

    suite, result_set = to_openeval(str(run_file), str(qrels_file), suite_id="msmarco-suite")
    assert validate_suite(suite).valid
    assert validate_result_set(result_set).valid
    tc1 = suite["test_cases"][0]
    assert tc1["retrieval_context"] == ["d2", "d1", "d3"]
