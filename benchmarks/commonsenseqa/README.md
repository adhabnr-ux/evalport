# CommonsenseQA

500 cases converted from the CommonsenseQA validation split into a
validated EvalPort suite.

## Source

- **Dataset**: [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) (`validation` split)
- **License**: MIT — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Talmor et al., 2019 — [CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge](https://arxiv.org/abs/1811.00937)

## What's in the suite

Five-way multiple-choice commonsense-reasoning questions sourced from
ConceptNet relations. Rendered as lettered multiple-choice; expected output
is the correct letter. Uses the `validation` split rather than `test`
because CommonsenseQA's public `test` split does not ship gold labels.

## Grader rationale

Single suite-level `exact_match` grader. Labeled-letter multiple choice
needs no fuzzy matching.

## Case count

500 (of 1,221 available in the validation split — capped by the pipeline's
default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../commonsenseqa/commonsenseqa.json
```
