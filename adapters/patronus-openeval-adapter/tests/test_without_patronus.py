"""Framework-free tests: these run with evalport-sdk alone (CI's min mode).

patronus is an optional extra, so the adapter module must import without it,
and from_openeval() on a suite with no Patronus-exported graders must not
need it. The real-object tests live in test_adapter.py, which skips itself
when patronus is absent.
"""
import importlib.util

import pytest

import patronus_openeval_adapter
from patronus_openeval_adapter import from_openeval, to_openeval

requires_no_patronus = pytest.mark.skipif(
    importlib.util.find_spec("patronus") is not None,
    reason="checks the missing-extra error path; patronus is installed",
)

SUITE = {
    "version": "1.0.0",
    "id": "hand-authored",
    "graders": [
        {"id": "g_custom", "type": "custom", "params": {"handler": "my.handler"}},
        {"id": "g_judge", "type": "llm_judge", "params": {"model": "m", "prompt": "Judge {output}"}},
    ],
    "test_cases": [
        {"id": "tc-1", "input": "Capital of Japan?", "expected_output": "Tokyo", "graders": ["g_custom"]},
        {"id": "tc-2", "input": "2+2?", "context": ["arithmetic"], "graders": ["g_judge"]},
    ],
}


def test_module_imports_without_patronus():
    assert callable(patronus_openeval_adapter.to_openeval)


def test_from_openeval_without_patronus_graders_needs_no_patronus():
    out = from_openeval(SUITE)
    assert out["ids"] == ["tc-1", "tc-2"]
    assert out["inputs"] == ["Capital of Japan?", "2+2?"]
    assert out["expected_outputs"] == ["Tokyo", None]
    assert out["contexts_list"] == [None, ["arithmetic"]]
    # Neither grader was exported by this adapter, so neither is reconstructed.
    assert out["evaluators"] == {}


@requires_no_patronus
def test_to_openeval_without_extra_names_the_extra():
    with pytest.raises(ImportError, match=r"patronus-openeval-adapter\[patronus\]"):
        to_openeval(inputs=["q"], evaluators={"e": object()})
