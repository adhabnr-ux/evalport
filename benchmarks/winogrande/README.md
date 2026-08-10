# WinoGrande — Pronoun Resolution

500 cases converted from the WinoGrande (`winogrande_xl`) validation split
into a validated EvalPort suite.

## Source

- **Dataset**: [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) (`winogrande_xl` config, `validation` split)
- **Original repo**: [allenai/winogrande](https://github.com/allenai/winogrande)
- **License**: CC-BY — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Sakaguchi et al., 2019 — [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641)

## What's in the suite

Fill-in-the-blank sentences with an underscore (`_`) standing in for a
pronoun/reference, and two candidate fills; the model must pick the one
that's contextually correct. Rendered as a two-option lettered
multiple-choice prompt; expected output is the correct letter. Rows whose
`answer` field wasn't `"1"` or `"2"` (a small number of malformed/unlabeled
rows) were skipped.

## Grader rationale

Single suite-level `exact_match` grader. Two-way labeled-letter choice needs
no fuzzy matching.

## Case count

500 (of 1,267 available in the `winogrande_xl` validation split — capped by
the pipeline's default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../winogrande/winogrande.json
```
