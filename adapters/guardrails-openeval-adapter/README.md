# guardrails-openeval-adapter

Convert [Guardrails AI](https://github.com/guardrails-ai/guardrails) `Guard` validation runs to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Install

```bash
pip install guardrails-openeval-adapter
```

## Usage

### Build a Guard, then export the values you're checking as an EvalPort suite

A Guardrails `Guard` validates one string at a time — there's no separate "input" and "output" the way a QA framework has. Attach every validator you want in a **single** `.use()` call:

```python
from guardrails import Guard, OnFailAction
from guardrails.validator_base import Validator, register_validator, PassResult, FailResult

@register_validator(name="myorg/valid-length", data_type="string")
class ValidLength(Validator):
    def __init__(self, min=0, max=1000, on_fail=None, **kwargs):
        super().__init__(on_fail=on_fail, min=min, max=max, **kwargs)
        self._min, self._max = min, max

    def validate(self, value, metadata):
        length = len(value)
        if self._min <= length <= self._max:
            return PassResult()
        return FailResult(error_message=f"length {length} not in [{self._min},{self._max}]")

# IMPORTANT: pass every validator to ONE .use() call. `.use(a).use(b)`
# silently replaces `a` with `b` instead of attaching both -- see
# "A footgun worth knowing about" below.
guard = Guard().use(ValidLength(min=1, max=50, on_fail=OnFailAction.NOOP))

from guardrails_openeval_adapter import to_openeval

suite = to_openeval(guard, ["hello world", "another candidate value"], ids=["a", "b"])

from openeval.validate import validate_suite
assert validate_suite(suite).valid
```

Every validator attached to `guard` becomes one EvalPort grader automatically — `to_openeval()` reads them straight off the Guard via `guard.get_validators()`, so there's no separate mapping to maintain. Each value you pass becomes one `TestCase` whose `input` is that string, graded by all of them (the same semantics as calling `guard.validate()`: every attached validator runs on every input).

### Import an EvalPort suite as validate()-ready values

```python
from guardrails_openeval_adapter import from_openeval

items = from_openeval(suite)  # -> [{"id": "a", "value": "hello world"}, {"id": "b", "value": "..."}]

outcomes, ids = [], []
for item in items:
    outcomes.append(guard.validate(item["value"]))
    ids.append(item["id"])
```

The id is returned alongside each value because `guard.validate()` gives back a `ValidationOutcome` with no reference to which test case it came from — you need to keep the pairing yourself, which is exactly what the loop above does.

### Export the validation outcomes as an EvalPort ResultSet

```python
from guardrails_openeval_adapter import evaluation_result_to_openeval

result_set = evaluation_result_to_openeval(guard, outcomes, ids, suite_id=suite["id"])

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every validator attached to `guard` becomes one `GraderResult` per test case. A validator that passed gets `score: 1.0`, `passed: true`. A validator that failed gets `score: 0.0`, `passed: false`, and the real Guardrails failure message preserved in `grader_result.reason`. A test case's overall `passed` is the AND of all of them.

### The full loop

```python
suite = to_openeval(guard, ["hi there", "no greeting here"], ids=["a", "b"])
items = from_openeval(suite)
outcomes = [guard.validate(item["value"]) for item in items]
ids = [item["id"] for item in items]
result_set = evaluation_result_to_openeval(guard, outcomes, ids, suite_id=suite["id"])
# result_set["results"][i]["test_case_id"] == suite["test_cases"][i]["id"], preserved end to end
```

## Why every grader is `"custom"`

A Guardrails validator is an arbitrary Python check. There's no reliable way to know from the outside whether a given validator means "exact string equality" (EvalPort's `exact_match`, the one grader type with no required `params`) versus a length check, a regex, a toxicity classifier, or a remote call to the Guardrails Hub. Rather than guess and risk misclassifying it, every validator maps to `custom`, with `params.handler` set to the validator's class name and `params` populated with its real constructor arguments (via the validator's own `get_args()`) — nothing fabricated.

## A real Guardrails quirk this adapter works *with*, not around

Confirmed directly against `guardrails-ai` 0.11.0: `ValidationOutcome.to_dict()["validationSummaries"]` only ever contains **failing** validators — a validator that passes produces no entry at all (this is Guardrails' own source, `ValidationSummary.from_validator_logs_only_fails()`, not something this adapter does). So `evaluation_result_to_openeval()` treats "not mentioned in `validationSummaries`" as a pass for that validator, on that test case — silence is success, exactly the way Guardrails itself treats it.

One consequence: if you attach **two instances of the same validator class** to one Guard (say, two separately-configured length checks), and both fail, Guardrails reports both failures — but neither entry carries anything beyond the shared class name, so this adapter can't safely tell which instance produced which failure. `to_openeval()` and `evaluation_result_to_openeval()` both raise `ValueError` up front if they detect this, rather than risk a silent misattribution. Register each distinct check as its own validator (a small `@register_validator` wrapper, same as the example above) to avoid it — this is also just better Guardrails practice; the official Hub validators are registered individually per check for the same reason.

## A footgun worth knowing about

Also confirmed directly: `Guard().use(a).use(b)` does **not** attach both validators — each `.use()` call *replaces* whatever was attached before, silently. Always attach every validator in one call: `Guard().use(a, b)` (or `Guard().use(validators=[a, b])`). This isn't specific to EvalPort — it's real Guardrails behavior — but it's easy to trip over, so it's called out here too.

## What round-trips losslessly, and what doesn't

Guardrails → EvalPort → Guardrails (via this adapter): the candidate strings and their ids round-trip exactly. The suite's `graders` fully describe each validator (class name + constructor args), but this adapter never reconstructs a live `Validator` from a suite — you re-attach your real validators to a `Guard` yourself, the same way every adapter in this ecosystem treats "the grader's actual implementation" as something the source framework runs, not something EvalPort re-executes.

Guardrails → EvalPort → some other tool: the candidate strings and each validator's pass/fail/score/reason are readable by any EvalPort consumer, but a different tool has no way to *run* a Guardrails-specific validator — the same tradeoff every adapter here takes for framework-specific logic with no native EvalPort equivalent.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
