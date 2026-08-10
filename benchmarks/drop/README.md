# DROP — Discrete Reasoning Over Paragraphs

500 cases converted from the DROP validation split into a validated
EvalPort suite.

## Source

- **Dataset**: [ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop) (`validation` split)
- **License**: CC-BY-SA-4.0 — see [../LICENSES.md](../LICENSES.md) (share-alike: redistributing this suite further requires carrying the same attribution)
- **Paper**: Dua et al., 2019 — [DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs](https://arxiv.org/abs/1903.00161)

## What's in the suite

Reading-comprehension questions that require actual discrete reasoning over
a passage — counting, sorting, addition/subtraction across multiple
passage facts — rather than single-span lookup, making it meaningfully
harder than plain extractive QA. Rows with no gold answer span were
skipped during conversion. `expected_output` is the first gold answer span.

## Grader rationale

Per-case inline `contains` graders (substring match, `ignore_case: true`)
— same rationale and same caveat as SQuAD 2.0: this is a lightweight proxy
for DROP's official F1 metric, not a reproduction of it (see
`metadata.evalport.grading_note` in the suite file, and
[`../squad2/README.md`](../squad2/README.md) for the fuller explanation of
why substring matching under-counts partial credit on span-extraction
tasks).

## Case count

500 (of 9,535 available in the validation split — capped by the pipeline's
default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../drop/drop.json
```
