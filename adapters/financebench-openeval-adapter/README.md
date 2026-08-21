# financebench-openeval-adapter

Convert [FinanceBench](https://github.com/patronus-ai/financebench) rows and scored model completions to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

FinanceBench is a benchmark for open-book financial question answering: 150 open-source, human-annotated questions about public companies' 10-K/10-Q/8-K/earnings filings, each with a gold answer and cited evidence text. This adapter is a straightforward fit for EvalPort's actual purpose (portable eval *datasets*) — it converts a real, static benchmark to/from a portable format, rather than wrapping a live evaluation SDK.

## Install

```bash
pip install "financebench-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/financebench-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (the same install path already verified working for this repo's other adapters).

FinanceBench itself is **not a Python package** — it's two JSONL files (`data/financebench_open_source.jsonl`, `data/financebench_document_information.jsonl`) plus a `results/` directory of JSONL files, all in the [patronus-ai/financebench](https://github.com/patronus-ai/financebench) repo. This adapter's only runtime dependency is `evalport-sdk`; there's no separate `[financebench]` extra to install because there's no upstream package to pin a version against.

## Usage

### Questions → EvalPort suite

```python
from financebench_openeval_adapter import load_jsonl, to_openeval
from openeval.validate import validate_suite

# Clone patronus-ai/financebench (or curl the raw files) first, then:
questions = load_jsonl("financebench/data/financebench_open_source.jsonl")
doc_info = load_jsonl("financebench/data/financebench_document_information.jsonl")

suite = to_openeval(questions, document_info_rows=doc_info, suite_id="financebench_open_source")
assert validate_suite(suite).valid

import json
with open("financebench_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

`document_info_rows` is optional — pass it to join in `doc_type` / `doc_period` / `doc_link` / `gics_sector` (from `financebench_document_information.jsonl`, matched by `doc_name`, the same join FinanceBench's own README documents) as `TestCase.metadata`. Omit it and every test case still converts, just without those extra metadata fields.

Each `TestCase.input` is the real question text; `expected_output` is FinanceBench's human-annotated gold answer; `context` is the cited evidence text(s); `metadata` carries every other FinanceBench field (`company`, `doc_name`, `question_type`, `question_reasoning`, `domain_question_num`, `justification`, `dataset_subset_label`, the full `evidence` list) under a `financebench.` prefix.

### Why the grader is `llm_judge`, not `exact_match`

FinanceBench answers are free-text financial statements (`"$1577.00 million"`, `"Increased by ~15% YoY"`) graded by human judgment in the original paper — `exact_match` would misgrade the large majority of correct-but-differently-formatted answers. `to_openeval()` emits one `llm_judge` grader with a prompt that tolerates numeric-formatting differences while still catching wrong numbers, refusals, or unsupported claims:

```python
suite = to_openeval(
    questions,
    judge_model="your-actual-judge-model",   # default "gpt-4o" is a placeholder, not a recommendation
    judge_prompt="...",                       # override the criteria if yours differ; must include {input}/{expected}/{output}
)
```

### EvalPort suite → FinanceBench rows (round trip)

```python
from financebench_openeval_adapter import from_openeval

rebuilt = from_openeval(suite)  # -> list[dict], same shape as financebench_open_source.jsonl rows
```

Round trips every field this adapter itself writes into `metadata` (prefixed `financebench.*`) back to its original FinanceBench key. Suites not originally produced by this adapter still convert — you just get `financebench_id`/`question`/`answer` back, with no fabricated values for fields that were never there.

### Scored results → EvalPort ResultSet

FinanceBench's `results/*.jsonl` files (one per evaluated model configuration, e.g. `results/gpt-4_sharedStore.jsonl`) already carry a **real human-annotated correctness label** per row — `"Correct Answer"`, `"Incorrect Answer"`, or `"Refusal"` (confirmed by loading a real results file; those are the only three values observed). `result_to_openeval()` carries that existing human judgment straight into EvalPort's `Result`/`GraderResult` shape — it does not re-grade anything with an LLM:

```python
from financebench_openeval_adapter import result_to_openeval
from openeval.validate import validate_result_set

model_results = load_jsonl("financebench/results/gpt-4_sharedStore.jsonl")
result_set = result_to_openeval(model_results, suite_id="financebench_open_source", run_id="gpt4_sharedstore_2023")
assert validate_result_set(result_set).valid
```

Each `Result.grader_results[0]` has `type: "human"` (score 1.0/passed=True only for `"Correct Answer"`; 0.0/False for `"Incorrect Answer"` and `"Refusal"`), `reason` set to the real label string, and `metadata` carrying `model_name` / `eval_mode` / `temp` / `gold_answer` from the source row. An unrecognized label value (should FinanceBench ever add one) is treated as `passed=False` and flagged with `metadata["financebench.unrecognized_label"] = True` rather than silently mis-scored or dropped.

## A note on FinanceBench's own README

FinanceBench's README documents the document-info sector column as `comany_sector_gics`. The real file on `main` (verified by downloading it for this adapter's test fixtures) actually uses `gics_sector`. This adapter reads the real field name and only falls back to the documented-but-unobserved one if present — so it keeps working whether or not that README typo ever gets fixed upstream.

## What round-trips losslessly, and what doesn't

- **Lossless**: `question`, `answer`, `company`, `doc_name`, `question_type`, `question_reasoning`, `domain_question_num`, `justification`, `dataset_subset_label`, `evidence` (the full nested list, including `evidence_text_full_page`).
- **Not part of the round trip by design**: `doc_type` / `doc_period` / `doc_link` / `gics_sector`, when joined in via `document_info_rows`, land in `TestCase.metadata` but `from_openeval()` deliberately excludes them from its output rows — those fields belong to the *separate* `financebench_document_information.jsonl` file, and reproducing them in the `financebench_open_source.jsonl` row shape would merge two files EvalPort's own suite metadata doesn't need to keep merged.

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

32 tests, all run against real data: `tests/fixtures/*.jsonl` are genuine rows trimmed from the real files at `https://raw.githubusercontent.com/patronus-ai/financebench/main/...` (not synthetic), chosen so the `financebench_id`/`doc_name` join and round-trip tests exercise real matching, not coincidence — and every suite/result-set produced in the tests is checked against the real `openeval.validate.validate_suite()` / `validate_result_set()`, not a mock.

## License

Apache 2.0, matching the rest of this repo.
