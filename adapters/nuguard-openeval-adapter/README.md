# nuguard-openeval-adapter

Convert [nuguard](https://github.com/NuGuardAI/nuguard) `Finding` /
`ValidateRunResult` objects to and from [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation results.

Built and tracked against [NuGuardAI/nuguard#355](https://github.com/NuGuardAI/nuguard/issues/355).
See [SPEC.md](SPEC.md) for the problem/solution/benefit writeup.

## Why a standalone package?

This follows the same playbook already used by the other adapters in this
repo (e.g. [crewai-openeval-adapter](../crewai-openeval-adapter)): it works
against nuguard's public `Finding`/`ValidateRunResult` shapes (pydantic model
instances, `.model_dump()` dicts, or plain dicts) from the outside, so this
gets you EvalPort export today with zero changes to nuguard core.

## Install

```bash
pip install "nuguard-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/nuguard-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support.

## Usage

```python
from nuguard_openeval_adapter import to_openeval, from_openeval

# run_result is whatever ValidateRunner.run() returned (a ValidateRunResult),
# or a plain dict/object with the same shape.
result_set = to_openeval(run_result)

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid

import json
with open("nuguard_results.json", "w") as f:
    json.dump(result_set, f, indent=2)

# ...and the other direction: reconstruct nuguard-Finding-shaped dicts from
# an EvalPort ResultSet (lossy/partial — see the docstring on from_openeval).
findings = from_openeval(result_set)
```

By default, `to_openeval()` derives `Result.test_case_id` from a finding's
`chain_id` (falling back to `finding_id`), and treats `Result.passed` as
`True` only when nuguard's own post-hoc probe explicitly disproved the
finding (`verified is False`) — every other case, including an unverified
finding, is a failed check:

```python
suite_id = to_openeval(run_result, suite_id="my_custom_suite_id")["suite_id"]
```

A run with zero findings (`scan_outcome == "no_findings"`) still produces a
spec-valid `ResultSet` — a single synthetic passing `Result` records that
the scan ran and found nothing, since EvalPort's `results` field must be
non-empty.

`capability_map` and `policy_records` (nuguard-specific, no 1:1 EvalPort
equivalent) are carried through under `ResultSet.metadata.nuguard` rather
than force-fit into `Result`/`GraderResult`.

## What maps where

| nuguard field | EvalPort field |
|---|---|
| `Finding.chain_id` (or `finding_id`) | `Result.test_case_id` |
| `Finding.severity` | `GraderResult.score` (banded to `[0.0, 1.0]`) |
| `Finding.verified is False` | `Result.passed` / `GraderResult.passed` |
| `Finding.evidence_quote` / `evidence` / `description` | `GraderResult.reason` |
| `Finding.owasp_llm_ref` / `owasp_asi_ref` / `mitre_atlas_technique` / `policy_clauses_violated` | `GraderResult.metadata.tags` |
| `Finding.goal_type` | `GraderResult.grader_id` (`gr_<goal_type>`) |
| `ValidateRunResult.scan_outcome` | `ResultSet.metadata.nuguard.scan_outcome` |
| `ValidateRunResult.capability_map` | `ResultSet.metadata.nuguard.capability_map` (summarized) |
| `ValidateRunResult.policy_records` | `ResultSet.metadata.nuguard.policy_records_count` |

Full field-by-field rationale is in the docstrings in
[`src/nuguard_openeval_adapter/__init__.py`](src/nuguard_openeval_adapter/__init__.py).

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
