# EvalPort → MLCommons Croissant Tasks

[Croissant Tasks](https://github.com/mlcommons/croissant/tree/main/tasks) is the
MLCommons vocabulary for describing ML tasks in JSON-LD. It has three parts:

- a benchmark is a `croissant:TaskProblem`;
- a submission is a `croissant:TaskSolution` that `schema:isBasedOn` the problem;
- the scoring of a submission is a `croissant:EvaluationTask` whose
  `croissant:evaluationResults` are `(metric, value)` pairs.

`evalport_to_croissant_tasks.py` turns one EvalPort Suite plus one ResultSet
into a single JSON-LD document containing all three nodes. The conversion needs
only the standard library.

```bash
python evalport_to_croissant_tasks.py ../basic-suite.json ../results.json \
    --base-uri https://example.org/evals/ > task.jsonld
```

## Mapping

| EvalPort | Croissant Tasks |
|---|---|
| `Suite.id`, `Suite.name`, `Suite.description` | `cr:TaskProblem` `@id` (`<base>suite/<id>`), `schema:name`, `schema:description` |
| `Suite.test_cases` (the suite file) | `cr:TaskProblem` `cr:input` → `schema:Dataset` (`--suite-url` or a fragment IRI) |
| The system under test | `cr:TaskProblem` `cr:implementation` → `cr:ImplementationSpec`. This is the "Spec" a problem must leave open. |
| The expected output | `cr:TaskProblem` `cr:output` → `cr:OutputSpec` |
| `ResultSet.run_id` | `cr:TaskSolution` `@id` (`<base>run/<run_id>`) |
| `ResultSet.suite_id` | `schema:isBasedOn` → the problem. A mismatch with `Suite.id` is rejected. |
| `provider.model`, or else `runner.name`; `runner.version` | `cr:TaskSolution` `cr:implementation` → `schema:SoftwareApplication` (`schema:name`, `schema:softwareVersion`) |
| The ResultSet file | `cr:TaskSolution` `cr:output` → `schema:Dataset` (`--result-set-url` or a fragment IRI) |
| `ResultSet.started_at` | `cr:TaskSolution` `schema:dateCreated` |
| Per-case `passed` | `cr:EvaluationResult`s `test_cases/{total,passed,pass_rate,unscored}` |
| Per-grader `score` / `passed` | `cr:EvaluationResult`s `<grader_id>/{pass_rate,mean_score,scored_count,unscored}` |

### What is lost

A `cr:EvaluationResult` holds exactly one metric and one value, so the export
carries summaries only. It does not carry:

- **Per-test-case and per-attempt results.** The `cr:output` dataset link points
  at the ResultSet, which still has them.
- **`score: null` ("not verified", EvalPort Rule 6).** These are not averaged
  in as zeros and not counted as failures. Each grader's `pass_rate` and
  `mean_score` are computed over non-null scores only, and the null count is
  reported separately as `<grader_id>/unscored`. `test_cases/unscored` counts
  cases where every grader was null-scored. When a grader has no scored results
  at all, its rate metrics are omitted rather than reported as 0.
- **Grader definitions** (type, params, judge model and prompt), **error
  objects**, **`group`**, and **`isolation`**.

### Numeric values are typed decimals

In JSON-LD, a JSON number with a fractional part becomes `xsd:double`. The
`croissant:value` shape accepts `xsd:decimal`, `xsd:integer`, `xsd:string` or
`schema:QuantitativeValue`, but not `xsd:double`. The exporter therefore writes
fractions as `{"@value": "0.5", "@type": "xsd:decimal"}` and integers as plain
JSON integers. This is reported upstream as mlcommons/croissant#1054.

## Tests

```bash
pip install -e ../../sdk/python -r requirements.txt
pytest -p no:cacheprovider .
```

`test_croissant_tasks.py` does two things:

1. It checks the conversion itself: summary arithmetic, null-score handling,
   node linking, and the CLI.
2. It validates the output against the Croissant Tasks ontology and SHACL
   shapes, which are vendored unmodified in `vendor/`. It uses the same pyshacl
   call as upstream's `tasks/validator.py`, with RDFS inference.

The negative controls confirm that the shapes do reject a solution with no
`schema:isBasedOn`, a problem with no Spec, and a bare JSON float value.

Upstream's `TaskProblemShape` has a known malformed property shape ("Bug A" in
mlcommons/croissant#1025 and #1027). pyshacl rejects it for every
`TaskProblem`, including upstream's own `valid_problem.jsonld`. The test works
around it by applying the fix suggested in #1025 to an in-memory copy. It
asserts that the vendored block is still present, so the test fails loudly if
upstream changes the shapes. The workaround can be deleted once upstream fixes
the shape. CI runs these tests in `.github/workflows/croissant-example.yml`.
