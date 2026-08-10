# HellaSwag — Commonsense Sentence Completion

500 cases converted from the HellaSwag validation split into a validated
EvalPort suite.

## Source

- **Dataset**: [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) (`validation` split)
- **Original repo**: [rowanz/hellaswag](https://github.com/rowanz/hellaswag)
- **License**: MIT — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Zellers et al., 2019 — [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)

## What's in the suite

Each case gives a short scenario (an activity label plus a context
sentence) and four candidate endings; the model must pick the ending that's
the most plausible commonsense continuation. Rendered as lettered
multiple-choice with the correct ending's letter as `expected_output`. A
small number of rows with an empty/missing `label` field (the HellaSwag
test split famously ships without gold labels; this suite uses the
`validation` split specifically because it has them) were skipped.

## Grader rationale

Single suite-level `exact_match` grader. Labeled-letter multiple choice
needs no fuzzy matching.

## Case count

500 (of 10,042 available in the validation split — capped by the pipeline's
default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../hellaswag/hellaswag.json
```
