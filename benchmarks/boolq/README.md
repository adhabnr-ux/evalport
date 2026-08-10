# BoolQ — Yes/No Reading Comprehension

500 cases converted from the BoolQ validation split into a validated
EvalPort suite.

## Source

- **Dataset**: [google/boolq](https://huggingface.co/datasets/google/boolq) (`validation` split)
- **Original repo**: [google-research-datasets/boolean-questions](https://github.com/google-research-datasets/boolean-questions)
- **License**: CC-BY-SA-3.0 — see [../LICENSES.md](../LICENSES.md) (share-alike: redistributing this suite further requires carrying the same attribution)
- **Paper**: Clark et al., 2019 — [BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions](https://arxiv.org/abs/1905.10044)

## What's in the suite

Naturally-occurring yes/no questions paired with a Wikipedia passage that
answers them. The prompt includes the full passage, the question, and an
explicit instruction to answer with exactly "true" or "false"; the expected
output is `"true"` or `"false"` per the dataset's boolean `answer` field.

## Grader rationale

Single suite-level `exact_match` grader (`ignore_case: true, strip: true`).
A binary true/false answer needs nothing more than an exact match once case
and surrounding whitespace are normalized.

## Case count

500 (of 3,270 available in the validation split — capped by the pipeline's
default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../boolq/boolq.json
```
