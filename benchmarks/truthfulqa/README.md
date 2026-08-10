# TruthfulQA — Truthfulness

817 cases converted from the full TruthfulQA (`generation` config)
validation split into a validated EvalPort suite — this is the only suite
in the hub converted uncapped, since the full split is a manageable size.

## Source

- **Dataset**: [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) (`generation` config, `validation` split)
- **License**: Apache-2.0 — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Lin, Hilton & Evans, 2021 — [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958)

## A loading note

The un-namespaced legacy repo id `truthful_qa` no longer resolves cleanly on
the HF Hub (raises `HfUriError`); this suite loads the current namespaced
repo id `truthfulqa/truthful_qa` instead.

## What's in the suite

Open-ended questions specifically designed to elicit common human
misconceptions/falsehoods if a model just mimics popular (but false) belief.
`expected_output` is the dataset's `best_answer` field (falling back to the
first entry in `correct_answers` if `best_answer` is empty). The full list
of all acceptable correct answers is preserved under
`metadata.truthfulqa_all_correct` for anyone building a more thorough
grader, along with the question's topic under `metadata.truthfulqa_category`.

## Grader rationale

Single suite-level `semantic_similarity` grader (`threshold: 0.8`,
`text-embedding-3-small`) rather than `exact_match` or `contains` — these
are open-ended free-text answers, not a fixed short string or letter, so
exact/substring matching would produce false negatives for correct answers
phrased differently. This is explicitly flagged as a coarse proxy: the
TruthfulQA authors' own evaluation uses fine-tuned GPT-judge/GPT-info
classifiers, which this suite does not attempt to reproduce (see
`metadata.evalport.grading_note` in the suite file itself). Treat scores
from this suite as directional, not as the paper's official metric.

## Case count

817 (the full `generation`/`validation` split — not capped).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../truthfulqa/truthfulqa.json
```
