# PIQA — Physical Interaction QA

500 cases converted from the PIQA validation split into a validated
EvalPort suite.

## Source

- **Dataset**: [ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa) (`validation` split, loaded via `revision="refs/convert/parquet"` — see note below)
- **Original site**: [yonatanbisk.com/piqa](https://yonatanbisk.com/piqa/)
- **License**: AFL-3.0 — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Bisk et al., 2019 — [PIQA: Reasoning about Physical Commonsense in Natural Language](https://arxiv.org/abs/1911.11641)

## What's in the suite

Two-way multiple-choice questions about physical commonsense — given a
goal, pick which of two proposed solutions is more physically sensible.
Rendered as a two-option lettered prompt; expected output is the correct
letter.

## A naming collision worth flagging

There are two unrelated projects on GitHub both called "PIQA" — this suite
is Yonatan Bisk's *Physical Interaction QA* (the one described above), not
the separate *Phrase-Indexed Question Answering* project that shares the
same acronym. The license and source links above were verified against
Bisk's actual project after catching this collision during research — see
[../LICENSES.md](../LICENSES.md) for the full note.

## A loading note

HuggingFace deprecated dataset-loading-script support, which broke the
straightforward `load_dataset("ybisk/piqa", split="validation")` call (it
raises `RuntimeError: Dataset scripts are no longer supported`). This suite
was converted using HF's auto-converted parquet mirror instead
(`revision="refs/convert/parquet"`), which serves the identical data without
requiring a loading script.

## Grader rationale

Single suite-level `exact_match` grader. Two-way labeled-letter choice needs
no fuzzy matching.

## Case count

500 (of 1,838 available in the validation split — capped by the pipeline's
default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../piqa/piqa.json
```
