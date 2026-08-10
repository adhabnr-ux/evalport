# MBPP — Mostly Basic Python Problems

300 cases converted from the MBPP (`full` config) test split into a
validated EvalPort suite.

## Source

- **Dataset**: [google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp) (`full` config, `test` split)
- **Original repo**: [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)
- **License**: CC-BY-4.0 — see [../LICENSES.md](../LICENSES.md)
- **Paper**: Austin et al., 2021 — [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732)

## What's in the suite

Short, crowd-sourced Python programming problems (a natural-language
description plus a handful of `assert`-based test cases). Like HumanEval,
each case carries its own inline `code`-type grader rather than a shared
suite-level one, since each problem's assertion set is unique — the suite-
level `graders` array is intentionally empty; see
[`../humaneval/README.md`](../humaneval/README.md) for the fuller
explanation of why inline graders are required here (the EvalPort spec's
`DANGLING_REFERENCE` check only applies to string-type references, and
per-case-varying test harnesses can't live in a single shared grader
definition).

The prompt shown to the model includes the natural-language task
description plus the literal assertion statements it must satisfy (MBPP's
task format gives the model the test cases up front, unlike HumanEval).
`params.source` for each grader is `{{completion}}` followed by the
problem's asserted test list; `metadata.mbpp_reference_code` preserves the
dataset's own reference solution.

## Grader rationale

Per-case inline `code` graders (`language: "python"`) — same rationale as
HumanEval: only actual execution against real assertions can verify
functional correctness of generated code. **Running this suite requires a
runner with sandboxed Python execution**; without one, skip via the spec's
`unsupported_grader_type` mechanism.

## Case count

300 (of 500 available in the `full`/`test` split — capped by the pipeline's
default `--limit 300`).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../mbpp/mbpp.json
```
