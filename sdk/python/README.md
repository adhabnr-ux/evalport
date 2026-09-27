# openeval

Python SDK for EvalPort — The Open Evaluation Standard.

## Install

```bash
pip install evalport-sdk
```

The distribution is `evalport-sdk`; the import name is `openeval`. (The unrelated `openeval` project on PyPI is not this SDK.)

## Usage

### Validate from the command line (`evalport-validate`)

Installing the package puts an `evalport-validate` command on your `PATH`
(also runnable as `python -m openeval.cli`). It is not in evalport-sdk 1.3.1
or earlier on PyPI; until a newer release, install from a checkout with
`pip install ./sdk/python`.

```bash
evalport-validate examples/*.json                  # files or globs; ** recurses
evalport-validate evals/ --include '*.evalport.json'  # directories are searched recursively
evalport-validate --type resultset run.json        # skip auto-detection
evalport-validate --format github '**/*.evalport.json'  # GitHub Actions annotations
```

- `--type auto|suite|testcase|grader|resultset` (default `auto`): auto-detection
  looks at top-level keys -- `results` means a result set, `test_cases` (or
  `test_cases_file`) a suite, `input` a test case, `type` + `id` a grader. A
  file matching none of these is reported as `UNKNOWN_TYPE`.
- `--format text|github|json` (default `text`). `github` prints one
  `::error file=<file>,title=EvalPort::<path>: <message> [<code>]` workflow
  command per error; `json` prints `{"valid": bool, "files": [...]}`.
- `--include GLOB` (default `*.json`): filename filter used when a directory is given.
- Exit status: `0` all valid; `1` any document invalid, unreadable, or not
  JSON (reported as `READ_ERROR` / `NOT_FOUND` / `INVALID_JSON`); `2` usage
  error, including a path or glob that matches no files.

```text
$ evalport-validate examples/basic-suite.json bad.json
examples/basic-suite.json: valid (suite)
bad.json: invalid (resultset), 1 error
  $.results[0].grader_results[0].score: must be in [0,1] or null [OUT_OF_RANGE]
2 files checked: 1 valid, 1 invalid
```

The same command backs this repo's GitHub Action and pre-commit hook -- see
[docs/github-action.md](../../docs/github-action.md).

### Validate a suite

```python
from openeval.validate import validate_suite

result = validate_suite({
    "version": "1.0.0",
    "id": "my_suite",
    "graders": [{"id": "gr1", "type": "exact_match"}],
    "test_cases": [{"id": "tc1", "input": "Hello", "expected_output": "Hi", "graders": ["gr1"]}]
})
print(result.valid)  # True
```

### Convert from Promptfoo

```python
from openeval.convert import from_promptfoo

suite = from_promptfoo(promptfoo_config)
```

### Convert from DeepEval

```python
from openeval.converters_deepeval import from_deepeval

suite = from_deepeval(deepeval_export)
```

### Convert from Inspect AI

```python
from openeval.converters_inspect import from_inspect

suite = from_inspect(inspect_data)
```

### Convert from OpenAI Evals

```python
from openeval.converters_openai import from_openai_evals

suite = from_openai_evals(evals_data)
```

### Convert from / to CrewAI

```python
from openeval.converters_crewai import from_crewai, crewai_result_to_result_set

suite = from_crewai({"tasks": crewai_task_defs})
result_set = crewai_result_to_result_set(crewai_run_result, suite, run_id="run_001")
```

### Compute summary

```python
from openeval.convert import compute_summary, create_result_set

summary = compute_summary(results)
result_set = create_result_set(suite, results, "run_001")
```

## API

### Validation
- `validate_suite(doc)` → `ValidationResult`
- `validate_test_case(doc)` → `ValidationResult`
- `validate_grader(doc)` → `ValidationResult`
- `validate_result_set(doc)` → `ValidationResult`
- `validate_document(doc, type)` → `ValidationResult`

### Command line
- `evalport-validate` console script → `openeval.cli:main(argv=None)` → exit code (`0`/`1`/`2`)

### Conversion
- `from_promptfoo(config)` → `dict`
- `from_deepeval(data)` → `dict` (from `converters_deepeval`)
- `from_inspect(data)` → `dict` (from `converters_inspect`)
- `from_openai_evals(data)` → `dict` (from `converters_openai`)
- `from_crewai(data)` → `dict` (from `converters_crewai`)
- `crewai_result_to_result_set(crew_result, suite, run_id)` → `dict` (from `converters_crewai`)
- `compute_summary(results)` → `dict`
- `create_result_set(suite, results, run_id)` → `dict`

## License

Apache 2.0
