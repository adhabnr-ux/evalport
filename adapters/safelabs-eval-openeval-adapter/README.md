# safelabs-eval-openeval-adapter

Convert [safelabs-eval](https://github.com/AgentSafeLabs/safelabs-eval) (AgentSafeLabs' OWASP Agentic Security Initiative red-teaming/eval framework for AI agents) prompts and run results to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

Proposed and discussed on [AgentSafeLabs/safelabs-eval#1](https://github.com/AgentSafeLabs/safelabs-eval/issues/1). The mapping was confirmed by the maintainer ([@iamwaqarjaved](https://github.com/iamwaqarjaved)), who asked for it to live as a standalone adapter in EvalPort's own `adapters/` directory — the same pattern [`crewai-openeval-adapter`](../crewai-openeval-adapter) and [`langsmith-openeval-adapter`](../langsmith-openeval-adapter) use: it works against safelabs-eval's public `PromptEntry` / `EvalRecord` / `EvalResult` / `ScoringResult` shapes (objects or dicts) from the outside, so nothing needs to be merged into safelabs-eval itself, and it stays under this repo's maintenance rather than coupling to safelabs-eval's internal refactors.

## Install

```bash
pip install "safelabs-eval-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/safelabs-eval-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support.

## Mapping

Unlike a framework whose "run" and "test definition" collapse into one object, safelabs-eval has a genuine split between its **prompt library** (test definitions) and its **eval results** (run output), which maps directly onto EvalPort's own EvalSuite / ResultSet split:

| safelabs-eval | EvalPort | Notes |
|---|---|---|
| `PromptEntry.id` | `TestCase.id` | |
| `PromptEntry.prompt` | `TestCase.input` | |
| `PromptEntry.expected_behavior` | `TestCase.expected_output` | |
| `PromptEntry.category`, `.severity`, `.tags` | `TestCase.metadata.safelabs` | EvalPort has no first-class OWASP-ASI field |
| detector (`PromptInjectionDetector`, `JailbreakDetector`, `DataLeakageDetector`, `HallucinationDetector`, `ScopeViolationDetector`) | `custom` Grader, `params.handler = "safelabs:<eval_type>"` | one grader per distinct detector `eval_type` actually used |
| `EvalRecord` | `Result` | `response` carried through unchanged as `actual_output` |
| `EvalRecord.scoring_result` | `Result.grader_results[0]` | full `ScoringResult` (`reasoning`, `indicators`, `remediation_hint`, `confidence`) preserved under `metadata.safelabs` |
| `EvalResult` (`records`, `categories_run`) | `ResultSet` | |

### VerdictLevel → score / passed

Confirmed by the maintainer on the issue thread:

| `VerdictLevel` | `GraderResult.score` | `GraderResult.passed` |
|---|---|---|
| `PASS` | `1.0` | `true` |
| `FAIL` | `0.0` | `false` |
| `VULNERABLE` | `0.0` | `false` |
| `UNCERTAIN` | `null` | `false` |

`FAIL` and `VULNERABLE` collapse to the same numeric score — a partial score for `FAIL` would misrepresent safelabs-eval's detectors' binary semantics — and are distinguished only via `metadata.safelabs.verdict`. `UNCERTAIN` maps to `score: null` rather than `0.0`, per EvalPort spec Rule 6 ("not verified" is distinct from "verified failing"), mirroring how `giskard-openeval-adapter` treats Giskard's `ERROR`/`SKIP` as no verdict reached.

## Usage

### Exporting the prompt library as an EvalPort suite

```python
from safelabs.prompts import get_library
from safelabs_eval_openeval_adapter import prompts_to_suite

library = get_library()  # or library.by_category("ASI06"), etc.
suite = prompts_to_suite(library, suite_id="safelabs_owasp_asi")

from openeval.validate import validate_suite
assert validate_suite(suite).valid

import json
with open("safelabs_asi_suite.json", "w") as f:
    json.dump(suite, f, indent=2)
```

### Exporting a run's results as an EvalPort ResultSet

```python
import asyncio
from safelabs.runner import run_eval
from safelabs_eval_openeval_adapter import eval_result_to_resultset

async def my_agent(prompt: str) -> str:
    return "I cannot help with that."

eval_result = asyncio.run(run_eval(my_agent, categories=["ASI01", "ASI06"]))
result_set = eval_result_to_resultset(eval_result, suite_id="safelabs_owasp_asi", run_id="ci_run_42")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

### Generic dispatcher

`to_openeval(obj)` inspects `obj` and delegates to `prompts_to_suite()` (for a prompt collection) or `eval_result_to_resultset()` (for an `EvalResult`), matching the `to_openeval()`/`from_openeval()` naming convention every adapter in this repo uses. Prefer calling the specific function directly when you know which one you have.

### Importing an EvalPort suite into safelabs-eval

```python
from safelabs_eval_openeval_adapter import from_openeval

entries = from_openeval(some_third_party_suite)
# entries: list of dicts with id/category/severity/prompt/expected_behavior/tags —
# pass into PromptEntry(**e) yourself, or run them straight through a Scorer.
```

This is the direction the maintainer specifically noted as a goal for `from_openeval()`: someone else's EvalPort test cases can be run through safelabs-eval's own detectors, not just the OWASP ASI library exported out. A suite with no `metadata.safelabs` (i.e. one that didn't originate from this adapter) defaults `category` to `"ASI01"` and `severity` to `"medium"` — override these deliberately rather than relying on the default when running third-party test cases through safelabs-eval.

## Data integrity

Per `DATA_INTEGRITY_RULES.md` ("archive raw responses before scoring"), this adapter never becomes the sole retention point for raw response text — `EvalRecord.response` is carried through to `Result.actual_output` unchanged, on top of whatever archival safelabs-eval itself already does.

## Credit

Mapping proposed and confirmed on [AgentSafeLabs/safelabs-eval#1](https://github.com/AgentSafeLabs/safelabs-eval/issues/1) with [@iamwaqarjaved](https://github.com/iamwaqarjaved).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
