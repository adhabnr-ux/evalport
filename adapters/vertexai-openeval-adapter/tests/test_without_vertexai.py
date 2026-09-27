"""Framework-free tests: these run with evalport-sdk alone (CI's min mode).

google-cloud-aiplatform[evaluation] is an optional extra, so the adapter
module must import without it, and from_openeval() on a suite with no
Vertex-exported graders must not need it. The real-object tests live in
test_adapter.py, which skips itself when vertexai is absent.
"""
import importlib.util

import pytest

import vertexai_openeval_adapter
from vertexai_openeval_adapter import from_openeval, to_openeval

requires_no_vertexai = pytest.mark.skipif(
    importlib.util.find_spec("vertexai") is not None,
    reason="checks the missing-extra error path; vertexai is installed",
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
        {"id": "tc-2", "input": "2+2?", "graders": ["g_judge"]},
    ],
}


def test_module_imports_without_vertexai():
    assert callable(vertexai_openeval_adapter.to_openeval)


def test_from_openeval_without_vertex_graders_needs_no_vertexai():
    out = from_openeval(SUITE)
    assert out["ids"] == ["tc-1", "tc-2"]
    assert out["instances"] == [{"prompt": "Capital of Japan?", "reference": "Tokyo"}, {"prompt": "2+2?"}]
    # Neither grader was exported by this adapter, so neither is reconstructed.
    assert out["metrics"] == []


@requires_no_vertexai
def test_to_openeval_without_extra_names_the_extra():
    with pytest.raises(ImportError, match=r"vertexai-openeval-adapter\[vertexai\]"):
        to_openeval(instances=[{"prompt": "q"}], metrics=[object()])
