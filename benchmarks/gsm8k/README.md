# GSM8K — Grade School Math

500 cases converted from the GSM8K test split into a validated EvalPort suite.

## Source

- **Dataset**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (`main` config, `test` split)
- **Original repo**: [openai/grade-school-math](https://github.com/openai/grade-school-math)
- **License**: MIT — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Cobbe et al., 2021 — [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

## What's in the suite

Each test case is a grade-school-level math word problem. GSM8K's reference
solutions are worked-through chains of arithmetic ending in a line like
`#### 18` giving the final numeric answer; the conversion pipeline extracts
just that final numeric answer as `expected_output` and keeps the full
worked solution under `metadata.gsm8k_full_solution` for anyone who wants
it (e.g. for chain-of-thought grading experiments), though it isn't used by
the suite's own grader.

## Grader rationale

Single suite-level `exact_match` grader (`ignore_case: true, strip: true`).
GSM8K's ground truth is a single unambiguous number, so exact string match
on the final answer — after stripping whitespace and normalizing case — is
the correct grading strategy; no semantic or fuzzy matching is needed. This
means a model's raw completion needs to be reduced to just the final answer
before grading (e.g. via a runner-side answer-extraction step or a prompt
that asks for the answer as a final line) — this suite does not itself do
answer extraction.

## Case count

500 (of 1,319 available in the test split — capped by the pipeline's default
`--limit 500` for redistribution-size reasons; re-run the converter with a
higher `--limit` to include more).

## Quickstart

```bash
python3 -c "
import json, sys
sys.path.insert(0, '../../sdk/python')
from openeval.validate import validate_suite
suite = json.load(open('gsm8k.json'))
print(validate_suite(suite).valid)
"
```

Or validate every benchmark suite at once from `_tools/`:

```bash
cd ../_tools && python3 validate_all.py
```
