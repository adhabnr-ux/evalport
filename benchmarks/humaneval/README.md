# HumanEval — Python Code Generation

164 cases converted from the full HumanEval test split into a validated
EvalPort suite (the full split, uncapped — HumanEval only has 164 problems
total).

## Source

- **Dataset**: [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval) (`test` split)
- **Original repo**: [openai/human-eval](https://github.com/openai/human-eval)
- **License**: MIT — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Chen et al., 2021 — [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)

## What's in the suite

Each case is a Python function signature + docstring (with example
doctests); the model is expected to complete the function body. Unlike the
other suites in this hub, HumanEval's grading is inherently *per-case* —
every problem has its own unique unit-test harness — so each test case
carries its **own inline `code`-type grader** rather than referencing a
shared suite-level grader. This is a deliberate architecture choice: the
EvalPort spec's `DANGLING_REFERENCE` validation only fires for string-type
grader references in test cases, and a shared suite-level grader can't hold
per-case-varying parameters (each problem's `source` test harness is
different), so inline dict graders are the only valid way to express this.

Each grader's `params.source` is HumanEval's canonical test harness
(`check(candidate)` function + assertions) with a `{{completion}}`
placeholder marking where the model's completion should be spliced in
before execution. `params.entry_point` names the function the harness calls.
The canonical reference solution is preserved under
`metadata.humaneval_canonical_solution` for anyone who wants a sanity-check
baseline.

## Grader rationale

Per-case inline `code` graders (`language: "python"`), not
`exact_match`/`contains` — code generation correctness can only be verified
by actually running the candidate solution against real test cases, not by
comparing text. **Running this suite requires a runner with sandboxed
Python execution.** A runner without one should skip these test cases
cleanly via the EvalPort spec's `unsupported_grader_type` mechanism rather
than attempting to `eval()` untrusted model output directly — this is
called out explicitly in the suite's own
`metadata.evalport.grading_note`.

## Case count

164 (the full test split — HumanEval doesn't have more than this).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../humaneval/humaneval.json
```
