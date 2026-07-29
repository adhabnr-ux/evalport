from openeval.validate import validate_suite, validate_grader, validate_test_case, validate_result_set

def test_valid_suite():
    assert validate_suite({"version":"1.0.0","id":"s","graders":[{"id":"g1","type":"exact_match"}],"test_cases":[{"id":"tc1","input":"hi","graders":["g1"]}]}).valid

def test_empty():
    assert not validate_suite({"version":"1.0.0","id":"s","test_cases":[]}).valid

def test_grader():
    assert validate_grader({"id":"g1","type":"exact_match"}).valid
    assert not validate_grader({"id":"g1","type":"bad"}).valid

def test_tc():
    assert validate_test_case({"id":"tc1","input":"hi","graders":["g1"]}).valid
    assert not validate_test_case({"id":"tc1","graders":["g1"]}).valid

def test_rs():
    assert validate_result_set({"version":"1.0.0","suite_id":"s","run_id":"r","started_at":"2026-01-01T00:00:00Z","results":[{"test_case_id":"tc1","passed":True,"grader_results":[{"grader_id":"g1","type":"exact_match","score":1.0,"passed":True}]}]}).valid
