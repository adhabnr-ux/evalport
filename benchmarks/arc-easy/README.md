# ARC-Easy — AI2 Reasoning Challenge (Easy)

500 cases converted from the ARC-Easy test split into a validated EvalPort suite.

See also [`../arc-challenge/`](../arc-challenge/) for the harder split of the
same benchmark.

## Source

- **Dataset**: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (`ARC-Easy` config, `test` split)
- **License**: CC-BY-SA-4.0 — see [../LICENSES.md](../LICENSES.md) (share-alike: redistributing this suite further requires carrying the same attribution)
- **Paper**: Clark et al., 2018 — [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457)

## What's in the suite

Grade-school-level science multiple-choice questions with 2-5 answer
choices each. Each question is rendered as a lettered multiple-choice
prompt (A/B/C/... one per line) and the expected output is the single
correct letter. A small number of source rows with malformed answer keys
(the labeled correct answer not present among the listed choices, or fewer
than 2 choices) were skipped during conversion — the exact count is
recorded per-suite under `metadata.evalport.skipped_malformed_cases`.

## Grader rationale

Single suite-level `exact_match` grader. Multiple-choice-with-a-labeled-
correct-letter is exactly what `exact_match` is for: the model's answer
either matches the correct letter or it doesn't, no fuzzy matching needed
(assuming the runner prompts the model to answer with just a letter).

## Case count

500 (of 2,376 available in the ARC-Easy test split — capped by the
pipeline's default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../arc-easy/arc-easy.json
```
