# geobenchx-openeval-adapter

Convert [GeoBenchX](https://github.com/Solirinai/GeoBenchX) `Task`/`Solution`/`ScoreValues` objects to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

Proposed and approved in [Solirinai/GeoBenchX#3](https://github.com/Solirinai/GeoBenchX/issues/3).

## Why a standalone package?

Same reasoning discussed on the issue: this works against GeoBenchX's public `Task`/`Solution`/`Step`/`TaskSet` shapes (pydantic objects, or an equivalent dict) from the outside, so no change to GeoBenchX itself is needed. GeoBenchX also isn't published to PyPI and has no installable package metadata at its repo root (no `pyproject.toml`/`setup.py`, only `requirements.txt`), so this adapter has no hard runtime dependency on a `geobenchx` package — it duck-types against attribute/key access instead, the same pattern already used by [crewai-openeval-adapter](../crewai-openeval-adapter) and [autogen-openeval-adapter](../autogen-openeval-adapter).

## Install

```bash
pip install "geobenchx-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/geobenchx-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support.

## Usage

```python
from geobenchx_openeval_adapter import to_openeval, results_to_openeval, from_openeval

# task_set is a geobenchx.dataclasses.TaskSet (or any list of Task-shaped
# objects/dicts) loaded e.g. via TaskSet.read_from_file(...)
suite = to_openeval(task_set)

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("geobenchx_suite.json", "w") as f:
    json.dump(suite, f, indent=2)

# After scoring (geobenchx.evaluation.score_solutions_set() has populated
# match_score_LLM on each task), export the ResultSet:
result_set = results_to_openeval(task_set, run_id="run_2026_08_31", started_at="2026-08-31T00:00:00Z")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

# ...and the other direction: load an EvalPort suite as GeoBenchX Task dicts
task_dicts = from_openeval(suite)
from geobenchx.dataclasses import Task
tasks = [Task(**d) for d in task_dicts]
```

## Mapping

**`TestCase`** (one per `Task`)
- `id = task.task_ID`, `input = task.task_text`, `tags = [l.value for l in task.task_labels]`.
- `expected_tools` = the union of `step.function_name` across every `reference_solutions[*].steps`, sorted. A reference solution that is a single `reject_task()` step therefore becomes `expected_tools == ["reject_task"]` — a first-class, checkable expectation instead of a special case — and is also mirrored as `metadata.unsolvable: true`.
- `metadata.reference_solutions` carries every reference solution's steps (`function_name`/`arguments`/`comment`) verbatim, so nothing about *how* to solve the task is lost by reducing it to `expected_tools`.
- `metadata.reference_solution_description` carries `task.reference_solution_description` through unchanged.

**`Grader`**: one shared `gr_geobenchx_llm_judge`, `type="llm_judge"`, whose `params.prompt` embeds GeoBenchX's actual 0/1/2 `EVALUATION_TAXONOMY` text (copied verbatim from `geobenchx/evaluation.py`) so a reader of the exported suite knows what the score means without the GeoBenchX source repo.

**`Result` / `GraderResult`** (one `Result` per scored `Task`, i.e. `match_score_LLM is not None`)
- `test_case_id = task.task_ID`, `actual_output` = a readable `func(args)  # comment` rendering of `task.generated_solution`.
- `score = match_score_LLM.value / 2.0` (normalized to EvalPort's required `[0.0, 1.0]` range), `passed = (match_score_LLM == ScoreValues.MATCH)`, `reason = match_reasoning_LLM`.
- The original 0/1/2 score is preserved under `metadata.raw_score_0_1_2`, and `match_score_Human`/`match_reasoning_Human` (when present) under `metadata.human_score`/`metadata.human_reasoning` — nothing is lossy-collapsed by the 0–1 normalization.

## Citation

If you use GeoBenchX's benchmark data or evaluation results through this adapter, please cite the original paper (per [Solirinai/GeoBenchX#3](https://github.com/Solirinai/GeoBenchX/issues/3) and `CITATION.cff`), which this adapter also stamps into `EvalSuite.metadata["citation"]` and `ResultSet.metadata["citation"]` automatically:

> Krechetova, Varvara; Kochedykov, Denis. "GeoBenchX: Benchmarking LLMs in Agent Solving Multistep Geospatial Tasks." Proceedings of the 1st ACM SIGSPATIAL International Workshop on Generative and Agentic AI for Multi-Modality Space-Time Intelligence (GeoGenAgent '25), ACM, 2025, pp. 27–35. DOI: [10.1145/3764915.3770721](https://doi.org/10.1145/3764915.3770721)

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
