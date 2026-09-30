"""Tests for the EvalPort -> Croissant Tasks exporter.

The SHACL checks use the Croissant Tasks ontology and shapes vendored under
``vendor/`` (from mlcommons/croissant, see vendor/README.md) and the same
pyshacl call as that repo's ``tasks/validator.py``. They are skipped when
``pyshacl``/``rdflib`` are not installed.
"""
import copy
import importlib.util
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from evalport_to_croissant_tasks import main, to_croissant_tasks  # noqa: E402

SUITE = json.loads((ROOT / "examples" / "basic-suite.json").read_text())
RESULTS = json.loads((ROOT / "examples" / "results.json").read_text())

HAS_SHACL = importlib.util.find_spec("pyshacl") is not None and importlib.util.find_spec("rdflib") is not None
CR = "http://mlcommons.org/croissant/"


def _value(v):
    return float(v["@value"]) if isinstance(v, dict) else v


def _metrics(doc):
    ev = next(n for n in doc["@graph"] if n["@type"] == "croissant:EvaluationTask")
    return {r["croissant:metric"]: _value(r["croissant:value"]) for r in ev["croissant:evaluationResults"]}


# Upstream shape bug ("Bug A" in mlcommons/croissant#1025 and #1027): the
# TaskProblemShape "at least one Spec" constraint is an sh:property with no
# sh:path, which pyshacl rejects as a malformed PropertyShape for *every*
# TaskProblem, including croissant's own testdata/valid_problem.jsonld. We apply
# the fix suggested there (lift sh:or to the NodeShape, one full property shape
# per alternative) to a copy at test time; the vendored file is left unmodified.
_BUG_A = """  sh:property [
    sh:or (
      [ sh:path croissant:input ; sh:class croissant:InputSpec ]
      [ sh:path croissant:output ; sh:class croissant:OutputSpec ]
      [ sh:path croissant:implementation ; sh:class croissant:ImplementationSpec ]
    ) ;
    sh:minCount 1 ;
    sh:message "A TaskProblem must have at least one property (input, output, or implementation) that is a spec class (InputSpec, OutputSpec, or ImplementationSpec)."
  ] ;"""
_BUG_A_FIX = """  sh:or (
    [ sh:property [ sh:path croissant:input ; sh:qualifiedValueShape [ sh:class croissant:InputSpec ] ; sh:qualifiedMinCount 1 ] ]
    [ sh:property [ sh:path croissant:output ; sh:qualifiedValueShape [ sh:class croissant:OutputSpec ] ; sh:qualifiedMinCount 1 ] ]
    [ sh:property [ sh:path croissant:implementation ; sh:qualifiedValueShape [ sh:class croissant:ImplementationSpec ] ; sh:qualifiedMinCount 1 ] ]
  ) ;"""


def _patched_shapes():
    text = (HERE / "vendor" / "croissant-tasks-shapes.ttl").read_text()
    assert text.count(_BUG_A) == 1, "vendored shapes changed; re-check the Bug A workaround"
    return text.replace(_BUG_A, _BUG_A_FIX)


def _shacl(doc, tmp_path):
    from pyshacl import validate
    from rdflib import Graph

    path = tmp_path / "task.jsonld"
    path.write_text(json.dumps(doc))
    data = Graph().parse(str(path), format="json-ld")
    shapes = Graph().parse(data=_patched_shapes(), format="turtle")
    ont = Graph().parse(str(HERE / "vendor" / "croissant-tasks.ttl"), format="turtle")
    conforms, _, text = validate(data, shacl_graph=shapes, ont_graph=ont, inference="rdfs")
    return conforms, text, data


def test_inputs_are_valid_evalport():
    from openeval.validate import validate_result_set, validate_suite

    assert validate_suite(SUITE).valid
    assert validate_result_set(RESULTS).valid


def test_summary_metrics_from_example_run():
    m = _metrics(to_croissant_tasks(SUITE, RESULTS))
    assert m["test_cases/total"] == 2
    assert m["test_cases/passed"] == 1
    assert m["test_cases/pass_rate"] == 0.5
    assert m["test_cases/unscored"] == 0
    assert m["gr_exact/pass_rate"] == 0.5
    assert m["gr_exact/scored_count"] == 2
    assert m["gr_exact/unscored"] == 0


def test_null_scores_are_counted_not_folded_into_failures():
    rs = copy.deepcopy(RESULTS)
    failing = next(r for r in rs["results"] if not r["passed"])
    for gr in failing["grader_results"]:
        gr["score"] = None
        gr["passed"] = False
    m = _metrics(to_croissant_tasks(SUITE, rs))
    # One scored pass remains, so the grader's pass rate is 1.0 over 1 scored result,
    # and the unverified case is reported separately rather than as a failure.
    assert m["gr_exact/pass_rate"] == 1.0
    assert m["gr_exact/scored_count"] == 1
    assert m["gr_exact/unscored"] == 1
    assert m["test_cases/unscored"] == 1


def test_all_null_grader_omits_rates():
    rs = copy.deepcopy(RESULTS)
    for r in rs["results"]:
        r["passed"] = False
        for gr in r["grader_results"]:
            gr["score"] = None
            gr["passed"] = False
    m = _metrics(to_croissant_tasks(SUITE, rs))
    assert "gr_exact/pass_rate" not in m and "gr_exact/mean_score" not in m
    assert m["gr_exact/unscored"] == 2


def test_mismatched_suite_rejected():
    rs = dict(RESULTS, suite_id="some_other_suite")
    with pytest.raises(ValueError):
        to_croissant_tasks(SUITE, rs)


def test_links_solution_to_problem_and_evaluation_to_solution():
    doc = to_croissant_tasks(SUITE, RESULTS, base_uri="https://example.org/evals/")
    problem, solution, evaluation = doc["@graph"]
    assert problem["@id"] == "https://example.org/evals/suite/suite_qa_basic"
    assert solution["schema:isBasedOn"] == {"@id": problem["@id"]}
    assert evaluation["croissant:evaluatedSolution"] == {"@id": solution["@id"]}
    assert solution["schema:dateCreated"] == RESULTS["started_at"]


def test_cli_writes_jsonld(tmp_path, capsys):
    assert main([str(ROOT / "examples" / "basic-suite.json"), str(ROOT / "examples" / "results.json")]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [n["@type"] for n in doc["@graph"]] == [
        "croissant:TaskProblem",
        "croissant:TaskSolution",
        "croissant:EvaluationTask",
    ]


@pytest.mark.skipif(not HAS_SHACL, reason="pyshacl/rdflib not installed")
def test_output_conforms_to_croissant_tasks_shapes(tmp_path):
    conforms, text, data = _shacl(to_croissant_tasks(SUITE, RESULTS), tmp_path)
    assert conforms, text
    from rdflib import RDF, URIRef

    types = {str(o) for o in data.objects(None, RDF.type)}
    for cls in ("TaskProblem", "TaskSolution", "EvaluationTask", "EvaluationResult"):
        assert CR + cls in types
    assert URIRef(CR + "TaskSolution") in set(data.objects(None, RDF.type))


@pytest.mark.skipif(not HAS_SHACL, reason="pyshacl/rdflib not installed")
def test_problem_without_any_spec_is_rejected(tmp_path):
    """Negative control for the patched Bug A shape: a problem with no Spec fails."""
    doc = to_croissant_tasks(SUITE, RESULTS)
    problem = doc["@graph"][0]
    problem["croissant:implementation"] = {"@type": "schema:SoftwareApplication"}
    problem["croissant:output"] = {"@type": "schema:Dataset"}
    conforms, _, _ = _shacl(doc, tmp_path)
    assert not conforms


@pytest.mark.skipif(not HAS_SHACL, reason="pyshacl/rdflib not installed")
def test_float_values_are_typed_decimals(tmp_path):
    """A bare JSON float becomes xsd:double, which the EvaluationResult shape rejects."""
    doc = to_croissant_tasks(SUITE, RESULTS)
    ev = doc["@graph"][2]
    ev["croissant:evaluationResults"].append(
        {"@type": "croissant:EvaluationResult", "croissant:metric": "raw_float", "croissant:value": 0.25}
    )
    conforms, _, _ = _shacl(doc, tmp_path)
    assert not conforms


@pytest.mark.skipif(not HAS_SHACL, reason="pyshacl/rdflib not installed")
def test_shapes_catch_a_broken_export(tmp_path):
    """Negative control: dropping isBasedOn must make the SHACL check fail."""
    doc = to_croissant_tasks(SUITE, RESULTS)
    del doc["@graph"][1]["schema:isBasedOn"]
    conforms, _, _ = _shacl(doc, tmp_path)
    assert not conforms
