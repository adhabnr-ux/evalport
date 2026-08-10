# ARC-Challenge — AI2 Reasoning Challenge (Challenge)

500 cases converted from the ARC-Challenge test split into a validated
EvalPort suite.

See also [`../arc-easy/`](../arc-easy/) for the easier split of the same
benchmark.

## Source

- **Dataset**: [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) (`ARC-Challenge` config, `test` split)
- **License**: CC-BY-SA-4.0 — see [../LICENSES.md](../LICENSES.md) (share-alike: redistributing this suite further requires carrying the same attribution)
- **Paper**: Clark et al., 2018 — [Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge](https://arxiv.org/abs/1803.05457)

## What's in the suite

The "Challenge" split of ARC — questions that both a retrieval-based and a
word-co-occurrence baseline got wrong, making this the harder half of the
benchmark. Same question/answer shape as ARC-Easy: 2-5 lettered multiple-
choice answers, expected output is the correct letter. Rows with malformed
answer keys were skipped during conversion; the count is recorded under
`metadata.evalport.skipped_malformed_cases`.

## Grader rationale

Single suite-level `exact_match` grader, same rationale as ARC-Easy:
labeled-letter multiple choice needs no fuzzy matching.

## Case count

500 (of 1,172 available in the ARC-Challenge test split — capped by the
pipeline's default `--limit 500`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../arc-challenge/arc-challenge.json
```
