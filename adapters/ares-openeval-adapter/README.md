# ares-openeval-adapter

Convert [IBM/ares](https://github.com/IBM/ares) attack-goal and evaluation output to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

This started as [IBM/ares#563](https://github.com/IBM/ares/issues/563), a proposal to add `ares-openeval-adapter` as an ARES plugin under `experimental-plugins/`. The maintainer (`stefano81`) confirmed that's the right home for it and invited a draft PR there. I don't currently have write access to open a branch directly on `IBM/ares` (expected for an outside contributor without collaborator access), so rather than block on that, this package ships here first — built against ARES's real current shapes, packaged and tested exactly as it's meant to land in `experimental-plugins/ares-openeval-adapter/` — and is offered as a normal PR into `IBM/ares` from a fork, or pulled in as-is, whichever the maintainers prefer. See the issue thread for the full context.

It works against ARES's public shapes (dataclass or dict) from the outside — `ares.utils.ConnectorResponse`, and the goal/eval-result dicts produced by `AttackGoal.run()` and consumed by `AttackEval` — so it has no hard dependency on the `ares` package itself and needs nothing merged into ARES's core to be usable today.

## Install

```bash
pip install "ares-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/ares-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working, see below).

## Usage

### Export ARES attack goals to an EvalPort suite

```python
from ares.goals.file_attack_goals import FileAttackGoals
from ares_openeval_adapter import to_openeval_suite

goals_loader = FileAttackGoals(config={
    "type": "ares.goals.file_attack_goals.FileAttackGoals",
    "file_path": "assets/attack_goals.csv",
    "output_path": "assets/attack_goals.json",
})
goals = goals_loader.run()  # list[dict]: {"goal": ..., "label": ..., ...}

suite = to_openeval_suite(goals, suite_id="my_ares_run")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

### Export ARES evaluation results to an EvalPort result set

```python
from ares_openeval_adapter import to_openeval_resultset

# eval_results is whatever a concrete AttackEval.evaluate() (e.g. KeywordEval,
# LLMEval) returned, after validate_evaluation() has populated
# `attack_successful` on each item.
result_set = to_openeval_resultset(eval_results, suite_id="my_ares_run", run_id="run_001")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

`passed` on each `Result` reflects target-model **robustness**: `True` when the attack did *not* succeed (`attack_successful == "No"`), `False` when it did (`"Yes"`) or the connector errored. An `"Unknown"` judgment (ARES's `prediction == -1.0`, no evaluator verdict) is also treated as not-passed but keeps `score: null` and `metadata.attack_successful == "Unknown"` so a consumer that cares about the ARES-specific tri-state can distinguish it from an outright failed attack. ARES's own judgment mechanism (keyword match / LLM judge / trained classifier, depending on which `AttackEval` subclass produced the results) is represented as a single `custom` grader (`gr_ares_attack_eval`, `params.handler = "ares:attack_eval"`) rather than modeled as one of EvalPort's standard grader types, since it isn't one — this follows the spec's own custom-grader mechanism (`SPEC.md` §Grader Type System) for exactly this situation.

### Import an EvalPort suite as ARES goals

```python
from ares_openeval_adapter import from_openeval

goals = from_openeval(suite)  # list[dict]: {"goal": ..., "target": ..., "label": ..., "additional_fields": {...}}

import json
with open("assets/attack_goals.json", "w") as f:
    json.dump(goals, f)
# usable as FileAttackGoals(config={"file_path": "assets/attack_goals.json", "jsonl": False, ...})
```

## Grounded in

- `ares.utils.ConnectorResponse` / `ares.utils.Status` — `src/ares/utils/__init__.py`
- `ares.evals.attack_eval.AttackEval` (`validate_evaluation`, `interpret_prediction`) — `src/ares/evals/attack_eval.py`
- `ares.goals.attack_goal.AttackGoal` / `ares.goals.file_attack_goals.FileAttackGoals` — `src/ares/goals/attack_goal.py`, `src/ares/goals/file_attack_goals.py`

all as of [IBM/ares@main](https://github.com/IBM/ares) at the time this was written.

## Tests

```bash
cd adapters/ares-openeval-adapter
pip install -e ".[test]"
pytest -v
```

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
