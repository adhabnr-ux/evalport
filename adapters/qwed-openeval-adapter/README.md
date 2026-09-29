# qwed-openeval-adapter

Convert [QWED](https://github.com/QWED-AI/qwed-verification) **Verification
Context v1.0** documents to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable evaluation suites and results
([spec](https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md)).

QWED is the verification layer here. It checks LLM outputs (math, logic, SQL,
code, facts, and more) and records the verdict, the admission decision and the
evidence in a Verification Context document. This adapter only converts those
documents. It does not run, re-run or re-judge a verification. QWED lives at
[QWED-AI/qwed-verification](https://github.com/QWED-AI/qwed-verification)
(the Verification Context spec is in
[`spec/v1.0/`](https://github.com/QWED-AI/qwed-verification/tree/main/spec/v1.0),
truth vs. admission in
[ADR-003](https://github.com/QWED-AI/qwed-verification/blob/main/docs/adr/ADR-003-truth-vs-admission.md)).
Other QWED projects are listed under [QWED-AI](https://github.com/QWED-AI).

The design was agreed with the QWED maintainer in
[QWED-AI/qwed-verification#325](https://github.com/QWED-AI/qwed-verification/issues/325).
The package lives in the EvalPort repo. QWED has not adopted it.

## Install

Not published to PyPI. Install from source:

```
pip install "qwed-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/qwed-openeval-adapter"
```

The only required dependency is `evalport-sdk`. Add the `qwed` extra
(`qwed>=7.1.0,<8`, tested with 7.2.1) to also run QWED's own
`qwed_sdk.validate_document()` on every document. That check covers the JSON
Schema and the `proof_ref` commitment.

## Dependency boundary

The adapter reads only the versioned Verification Context v1.0 JSON document.
QWED's supported surfaces all produce it:

- `QWEDClient.create_verification_context_from_diagnostic(...)`, which calls HTTP `POST /verification-context/from-diagnostic`
- `qwed context from-diagnostic --diagnostic-file ... --query ... --verifier ...` (CLI)
- `qwed_sdk.VerificationContextDocument` (`.to_dict()`, or pass the object directly)

It never imports `qwed_new.*`. When `qwed` is installed, it imports only the
public `qwed_sdk.validate_document` re-export, and only inside the function
that uses it. A test checks that importing the adapter imports neither
`qwed_sdk` nor `qwed_new`.

The legacy `QWEDClient.verify_*` convenience results (`VerificationResult`)
are rejected with a `TypeError`. They carry no verdict, admission or
`proof_ref` to map. QWED tracks that separately in
[#327](https://github.com/QWED-AI/qwed-verification/issues/327), and the
adapter does not work around it.

## Usage

```python
from qwed_sdk import QWEDClient
from qwed_openeval_adapter import to_openeval, suite_to_openeval, from_openeval
from openeval.validate import validate_result_set

client = QWEDClient(api_key="qwed_...", base_url="http://localhost:8000")
doc = client.create_verification_context_from_diagnostic(
    diagnostic=diagnostic_result_dict, query="SELECT ...", verifier="sql",
)

result_set = to_openeval([doc], suite_id="my-sql-checks", test_case_ids=["orders-42"])
assert validate_result_set(result_set).valid

suite = suite_to_openeval([doc], suite_id="my-sql-checks", test_case_ids=["orders-42"])
docs_again = from_openeval(result_set)   # == [doc]
```

`to_openeval(..., validate=None)` runs `qwed_sdk.validate_document` when
`qwed` is installed and only the structural invariants otherwise.
`validate=True` requires `qwed`. `validate=False` runs the structural checks
only. Each `Result` records which check ran in
`metadata.qwed.vc_validation`, either `"qwed_sdk.validate_document"` or
`"structural"`. The structural check does not re-derive the `proof_ref`
commitment.

## Mapping

Each Verification Context document becomes one EvalPort `Result` with **two
independent `GraderResult`s**. The verdict and the admission decision are
never merged.

| grader_id | type | QWED value | `score` | `passed` |
|---|---|---|---|---|
| `qwed.verdict` | `qwed_verification_context` | `verdict: VERIFIED` | `1.0` | `true` |
| `qwed.verdict` | `qwed_verification_context` | `verdict: UNVERIFIABLE` or `BLOCKED` | `null` | `false` |
| `qwed.admission` | `qwed_admission_decision` | `context.decision.admission: ADMIT` | `1.0` | `true` |
| `qwed.admission` | `qwed_admission_decision` | `context.decision.admission: DENY` | `0.0` | `false` |

What `passed` means on each grader:

- **Verdict grader:** `passed` means the claim was definitively established by
  QWED (`VERIFIED`). It does not mean the artifact is safe or good.
- **Admission grader:** `passed` means the result is admissible (`ADMIT`).
- **`Result.passed`** is `verdict == VERIFIED and admission == ADMIT`. It
  equals EvalPort's default `all` aggregation over the non-null-scored graders.

**`VERIFIED` + `DENY`** is the reason the two graders stay separate. It is
ADR-003's "VERIFIED-as-unsafe" case: QWED proved the code is unsafe, so the
verdict grader is `score 1.0, passed true` and the admission grader is
`score 0.0, passed false`. The `Result` is `passed: false`. Collapsing both
into one `passed` flag would either hide that the danger was proven or admit
proven-unsafe code.

**`UNVERIFIABLE` and `BLOCKED` are `score: null`, not `0.0`.** EvalPort's
Validation Rule 6 says "`score: null` means 'not verified'" and requires
`passed: false` alongside it. It also says consumers "MUST NOT treat a
`null`-score result as equivalent to a scored failure when computing pass
rates, aggregate statistics, or the suite-level `passed` field". That matches
QWED's fail-closed "not proven". Verification Context v1.0 has no "refuted"
verdict, so the verdict grader never emits `0.0`. The admission grader is
still scored in these cases. QWED's schema requires `DENY` for
`UNVERIFIABLE`/`BLOCKED`, and that is a definitive decision.

**Scores are numbers.** `GraderResult.score` is `number | null` in [0, 1]
(Validation Rule 5, `spec/schemas/resultset.json`). QWED's own strings never
go in `score`. They are kept verbatim here:

- `GraderResult.metadata.qwed.verdict` and `GraderResult.metadata.qwed.admission`
- `Result.metadata.qwed`: `verdict`, `admission`, `verifier`, `verifier_version`, `proof_ref`, `spec_version`, `vc_validation`
- `GraderResult.reason`: a readable sentence that includes the evidence's `agent_message` when there is one

**The full Verification Context is preserved.** The complete document, with
its `object` and all four `context` layers (interpretation, proof, evidence,
decision), is embedded as a deep copy under the verdict grader's
`metadata.qwed.verification_context`. `proof_ref` therefore stays resolvable
against its bound payload. `from_openeval()` returns those documents
unchanged. It raises `ValueError` if a `Result`'s scores or `passed` values no
longer match the embedded document, for example after an edited score.

`suite_to_openeval()` builds an `EvalSuite` with one `TestCase` per document.
The `input` is `object.formalization.source_query`, or `formal_statement`
when that is missing. Both graders are defined once at suite level with
`params.handler`, since they are non-standard types.

### Correction to the earlier design in #325

An earlier version of this mapping, in the comment the maintainer approved on
#325, put QWED's strings in `GraderResult.score` (`"VERIFIED"`, `"DENY"`,
etc.). It justified that by saying EvalPort's `score` is a string. That was
wrong: `score` is `number | null` in [0, 1], and that shape fails both
`validate_result_set()` and the raw JSON Schema. This package keeps the
approved structure: two independent graders, `qwed.verdict` and
`qwed.admission`, `Result.passed` computed from both, and the full
Verification Context preserved. Only the score encoding changes, as shown in
the table above. The regression tests
`test_regression_old_string_score_design_fails_*` pin this.

## Tests

```
pip install -e ".[test]"   # pytest, jsonschema, qwed
pytest tests
```

The fixtures in `tests/fixtures/` were generated with qwed 7.2.1 by
`tests/fixtures/make_fixtures.py`:

- The two `VERIFIED` fixtures were built with the public
  `VerificationContextDocument.verified(...)` constructor, which computes the
  real `proof_ref`. They are constructed documents, not engine output.
- The `UNVERIFIABLE` and `BLOCKED` fixtures are real output of
  `qwed context from-diagnostic`.

Tests that need `qwed` or `jsonschema` skip themselves when those are missing,
so EvalPort's min-mode CI (`evalport-sdk` only) still runs every
framework-free test.

## Spec

EvalPort specification:
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

QWED Verification Context v1.0 specification:
https://github.com/QWED-AI/qwed-verification/blob/main/spec/v1.0/verification-context.md

## License

Apache 2.0, see LICENSE.
