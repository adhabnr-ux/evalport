# agent-skills-eval-openeval-adapter

Convert the on-disk run artifacts written by
[agent-skills-eval](https://github.com/darkrishabh/agent-skills-eval) —
`grading.json`, `timing.json`, `outputs/response.txt` — to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

## Why a standalone package, and why it reads files instead of importing agent-skills-eval

agent-skills-eval is an npm package (TypeScript CLI + SDK) with no Python API,
so there is nothing to `pip install` from it directly. This adapter instead
reads the JSON artifacts it writes to its `workspace` directory, per the
layout documented in agent-skills-eval's own
[`docs/artifact-contract.md`](https://github.com/darkrishabh/agent-skills-eval/blob/main/docs/artifact-contract.md).

Built at the maintainer's explicit direction: see
[agent-skills-eval#34](https://github.com/darkrishabh/agent-skills-eval/issues/34#issuecomment-5888398803)
("A standalone adapter in EvalPort is the direction we'd prefer... No
first-party export or dependency is planned at this stage.").

Pinned against agent-skills-eval 0.1.1 plus the (at the time of writing)
unreleased `docs/artifact-contract.md` / `src/grade.ts` changes merged in
[PR #37](https://github.com/darkrishabh/agent-skills-eval/pull/37) (commit
`72c3c65`). Unknown/additive fields in `grading.json` / `timing.json` are
ignored, not rejected, so a later agent-skills-eval release that only adds
fields keeps working.

## Install

```
pip install "agent-skills-eval-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/agent-skills-eval-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support.

## Usage

### From a workspace directory (typical case)

```python
from agent_skills_eval_openeval_adapter import iter_eval_dirs, load_run_artifacts, to_openeval
from openeval.validate import validate_result_set

runs = [load_run_artifacts(d) for d in iter_eval_dirs("./agent-skills-workspace")]

# One ResultSet per mode — agent-skills-eval runs with_skill and, when
# `baseline: true`, without_skill, and the two shouldn't be blended into one
# ResultSet (see to_openeval()'s docstring on why `mode` is declared once).
with_skill = to_openeval(
    [r for r in runs if "with_skill" in r["metadata"]["source_path"]],
    suite_id="basic-skill",
    run_id="run-2026-09-29",
    mode="with_skill",
)
assert validate_result_set(with_skill).valid
```

### From individual run artifacts

```python
from agent_skills_eval_openeval_adapter import to_openeval

result_set = to_openeval(
    [
        {
            "test_case_id": "case-1",
            "grading": {"assertion_results": [...], "summary": {...}},  # parsed grading.json
            "actual_output": "...",                                    # outputs/response.txt
            "duration_ms": 842,                                        # timing.json
        },
    ],
    suite_id="basic-skill",
    run_id="run-1",
    mode="with_skill",
)
```

### Classifying assertion rows correctly (important)

`grading.json`'s `assertion_results` rows don't carry a type — they're a
positional concatenation of LLM-judge rubric results followed by
deterministic tool-assertion results (see `gradeOutputs()` in
agent-skills-eval's `src/grade.ts`). Pass the original eval definition as
`eval_def` to classify each row exactly:

```python
result_set = to_openeval(
    [
        {
            "test_case_id": "case-1",
            "grading": grading_json,
            "eval_def": {
                "assertions": skill_eval["assertions"],
                "tool_assertions": skill_eval.get("tool_assertions", []),
            },
        },
    ],
    suite_id="basic-skill",
    run_id="run-1",
    mode="with_skill",
)
```

Without `eval_def`, every row is reported as EvalPort grader type `custom`
(never guessed as `llm_judge`) with `metadata.kind_inferred = False`, per
[the upstream maintainer's explicit request](https://github.com/darkrishabh/agent-skills-eval/issues/34#issuecomment-5888398803).

### Round-tripping back

```python
from agent_skills_eval_openeval_adapter import from_openeval

by_test_case = from_openeval(result_set)
by_test_case["case-1"]["grading"]  # {"assertion_results": [...], "summary": {...}}
```

`text` / `passed` / `evidence` round-trip exactly for rows this adapter
produced. EvalPort-side bookkeeping (`score`, `kind`, `kind_inferred`) has no
field in agent-skills-eval's own `AssertionResult` type and isn't written
back — a documented, one-way loss, not a silent one.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 — see LICENSE.
