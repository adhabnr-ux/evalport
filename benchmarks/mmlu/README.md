# MMLU — Massive Multitask Language Understanding

6 subject suites converted from MMLU's `test` split into validated EvalPort
suites, covering a deliberately varied slice of MMLU's 57 subjects rather
than the full set (57 separate suites was judged more than this hub needs
to demonstrate the format — see "Why only 6 subjects" below).

## Source

- **Dataset**: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) (per-subject config, `test` split)
- **Original repo**: [hendrycks/test](https://github.com/hendrycks/test)
- **License**: MIT — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Hendrycks et al., 2020 — [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)

## Subjects included

| File | Subject | Cases |
|---|---|---|
| `mmlu-high-school-mathematics.json` | High School Mathematics | 200 |
| `mmlu-high-school-us-history.json` | High School US History | 200 |
| `mmlu-high-school-computer-science.json` | High School Computer Science | 100 |
| `mmlu-professional-law.json` | Professional Law | 200 |
| `mmlu-college-biology.json` | College Biology | 144 |
| `mmlu-moral-scenarios.json` | Moral Scenarios | 200 |

Each is a fully independent, individually-valid EvalPort suite (own `id`:
`bench_mmlu_<subject>`) — pick and run whichever subjects are relevant
rather than needing all 6.

## Why only 6 subjects

MMLU ships 57 subject configs total. This hub includes 6 spanning a
deliberately varied difficulty/domain range (STEM, humanities, professional,
and — notably — the qualitatively different "Moral Scenarios" ethics-judgment
format) as a representative demonstration rather than exhaustively
converting all 57, which would be low-marginal-value repetition of the exact
same conversion logic. Add more subjects by adding to `MMLU_SUBJECTS` in
`../_tools/convert_hf_dataset.py` and re-running `convert_hf_dataset.py mmlu`
— the conversion code already handles any of the 57 valid subject configs.

## What's in each suite

Four-way multiple-choice academic questions. Rendered as lettered
multiple-choice; `expected_output` is the correct letter (MMLU's `answer`
field is already a 0-indexed integer into `choices`, mapped directly to a
letter).

## Grader rationale

Single suite-level `exact_match` grader per subject. Labeled-letter
multiple choice needs no fuzzy matching.

## Case count

200 per subject except High School Computer Science (100, the full test
split size for that subject) and College Biology (144, likewise its full
test split size) — the pipeline's default `--limit 200` naturally caps at
the split size when the split itself is smaller.

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../mmlu/mmlu-high-school-mathematics.json
# or validate the whole hub at once:
cd ../_tools && python3 validate_all.py
```
