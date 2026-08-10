# BIG-Bench Hard (BBH)

3 task suites converted from BIG-Bench Hard into validated EvalPort suites
— a deliberately varied slice of BBH's 23 tasks (see "Why only 3 tasks"
below).

## Source

- **Dataset**: [lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh) (per-task config, `test` split)
- **Original repo**: [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)
- **License**: MIT — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Suzgun et al., 2022 — [Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them](https://arxiv.org/abs/2210.09261)

## Tasks included

| File | Task | Cases |
|---|---|---|
| `bbh-logical-deduction-five-objects.json` | Logical Deduction (five objects) | 250 |
| `bbh-causal-judgement.json` | Causal Judgement | 187 |
| `bbh-date-understanding.json` | Date Understanding | 250 |

Each is a fully independent, individually-valid EvalPort suite (own `id`:
`bench_bbh_<task>`).

## Why only 3 tasks

BBH ships 23 tasks total, all originally drawn from the larger BIG-Bench
suite as ones where prior language models performed at or below random
chance. This hub includes 3 spanning different reasoning styles — ordering/
constraint-satisfaction logic (Logical Deduction), causal reasoning about
natural-language scenarios (Causal Judgement), and date arithmetic (Date
Understanding) — as a representative sample rather than converting all 23,
which is the same conversion logic repeated 20 more times for limited
additional demonstration value. Add more tasks by adding to `BBH_TASKS` in
`../_tools/convert_hf_dataset.py` and re-running
`convert_hf_dataset.py bbh`.

## What's in each suite

Each task's own native format — Logical Deduction and Date Understanding
are multiple-choice with the answer given in the dataset's own
`(X)`-lettered format; Causal Judgement is a yes/no causal-reasoning
question. `input` is the task's raw prompt (BBH prompts are already
fully-formed natural-language task instructions); `expected_output` is the
dataset's own `target` field verbatim, unmodified — this suite does not
reformat BBH's native answer format into a separate multiple-choice
rendering the way ARC/MMLU/HellaSwag/etc. do, since BBH's own prompts
already include any needed answer-choice framing per task.

## Grader rationale

Single suite-level `exact_match` grader per task (`ignore_case: true,
strip: true`). BBH's `target` field is always a single unambiguous short
answer string in the dataset's own canonical format, so exact match after
whitespace/case normalization is the correct grading strategy — same as
GSM8K and BoolQ.

## Case count

250 for Logical Deduction and Date Understanding (each task's full test
split size), 187 for Causal Judgement (its full test split size — smaller
than the pipeline's `--limit 250` default, so no cases were dropped).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../bbh/bbh-causal-judgement.json
# or validate the whole hub at once:
cd ../_tools && python3 validate_all.py
```
