"""Framework-free tests: these run with evalport-sdk alone (CI's min mode).

arthur-bench is an optional extra, so the adapter module itself must import
without it, and to_openeval()/run_to_openeval() -- which only read attributes
off the objects they're given -- must work against stand-ins with the same
public shape as arthur_bench's TestSuite/TestRun. The real-object tests live
in test_adapter.py, which skips itself when arthur-bench is absent.
"""
import importlib.util
from types import SimpleNamespace

import pytest
from openeval.types import OPENEVAL_VERSION
from openeval.validate import validate_result_set, validate_suite

import arthur_bench_openeval_adapter
from arthur_bench_openeval_adapter import from_openeval, run_to_openeval, to_openeval

requires_no_arthur_bench = pytest.mark.skipif(
    importlib.util.find_spec("arthur_bench") is not None,
    reason="checks the missing-extra error path; arthur-bench is installed",
)


class StubScorer:
    """Same public surface arthur_bench.scoring.scorer.Scorer exposes here."""

    def __init__(self, name, config=None):
        self._name = name
        self._config = config or {}

    def name(self):
        return self._name

    def to_dict(self, warn=False):
        return dict(self._config)


def _stub_suite():
    return SimpleNamespace(
        name="capitals",
        description="Two capital-city questions",
        scorer=StubScorer("exact_match", {"case_sensitive": True}),
        test_cases=[
            SimpleNamespace(id="tc-1", input="Capital of Japan?", reference_output="Tokyo"),
            SimpleNamespace(id="tc-2", input="Capital of France?", reference_output=None),
        ],
    )


def _stub_run():
    match = SimpleNamespace(name="match", description="exact match")
    no_match = SimpleNamespace(name="no_match", description="no exact match")
    return SimpleNamespace(
        id="run-1",
        test_suite_id="suite-1",
        model_name="stub-model",
        foundation_model=None,
        prompt_template=None,
        model_version=None,
        test_cases=[
            SimpleNamespace(id="tc-1", score=1.0, output="Tokyo", score_result=SimpleNamespace(category=match)),
            SimpleNamespace(id="tc-2", score=0.0, output="Lyon", score_result=SimpleNamespace(category=no_match)),
        ],
    )


def test_module_imports_and_stamps_the_sdk_spec_version():
    assert arthur_bench_openeval_adapter.SPEC_VERSION == OPENEVAL_VERSION


def test_to_openeval_duck_typed_suite_is_valid():
    suite = to_openeval(_stub_suite())
    result = validate_suite(suite)
    assert result.valid, result.errors
    assert suite["version"] == OPENEVAL_VERSION
    assert suite["id"] == "capitals"
    assert suite["graders"][0]["id"] == "exact_match"
    assert suite["graders"][0]["params"] == {"handler": "exact_match", "config": {"case_sensitive": True}}
    assert suite["test_cases"][0]["expected_output"] == "Tokyo"
    assert "expected_output" not in suite["test_cases"][1]
    assert suite["metadata"] == {"arthur_bench": {"description": "Two capital-city questions"}}


def test_run_to_openeval_duck_typed_run_is_valid():
    rs = run_to_openeval(_stub_run(), scorer_name="exact_match")
    result = validate_result_set(rs)
    assert result.valid, result.errors
    assert rs["version"] == OPENEVAL_VERSION
    assert [r["passed"] for r in rs["results"]] == [True, False]
    assert rs["summary"] == {"total": 2, "passed": 1, "failed": 1, "pass_rate": 0.5}
    assert rs["metadata"] == {"arthur_bench": {"model_name": "stub-model"}}


@requires_no_arthur_bench
def test_from_openeval_without_extra_names_the_extra():
    suite = to_openeval(_stub_suite())
    with pytest.raises(ImportError, match=r"arthur-bench-openeval-adapter\[arthur-bench\]"):
        from_openeval(suite)
