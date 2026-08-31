# nuguard-openeval-adapter: problem, solution, benefit

Posted per the request on [NuGuardAI/nuguard#355](https://github.com/NuGuardAI/nuguard/issues/355)
("Go ahead with the first version of this adapter, open the PR against
develop. Create a spec that describes the problem, solution, user benefits.").

## Problem

nuguard's `validate` mode produces a `ValidateRunResult`
(`nuguard/models/validate.py`) carrying structured `Finding`s
(`nuguard/models/finding.py`) with real substance: `severity`, `evidence`,
OWASP/MITRE ATLAS references, an NGRS risk score, and a post-hoc `verified`
flag. That result shape is specific to nuguard, so it can't be consumed by
any tool downstream that speaks [EvalPort](https://github.com/adhabnr-ux/evalport)
(a portable, framework-agnostic result format already implemented by 30+
adapters — DeepEval, Promptfoo, CrewAI, AutoGen, Braintrust, MLflow, and
others) without a bespoke, one-off conversion every time.

## Solution

A standalone `nuguard-openeval-adapter` package — no changes to nuguard
core — that reads nuguard's public `Finding`/`ValidateRunResult` shapes
(pydantic model, `.model_dump()` dict, or plain dict) and produces an
`openeval.validate.validate_result_set()`-valid EvalPort `ResultSet`:

- **One `Finding` → one `Result`.** `chain_id` (falling back to
  `finding_id`) becomes `test_case_id`; `severity` bands to EvalPort's
  `[0.0, 1.0]` score axis; `evidence_quote`/`evidence`/`description` becomes
  `reason`; OWASP/MITRE/policy-clause references become tags in
  `GraderResult.metadata`.
- **A `Finding`'s existence is the failure signal.** `Result.passed` is
  `True` only when nuguard's own post-hoc probe explicitly disproved it
  (`verified is False`) — everything else (unverified or reproduced) is a
  failed check, which matches how a scan finding is actually used
  downstream (as something to fix, not something to celebrate).
- **`scan_outcome` rolls up to `ResultSet.summary`.** A run with zero
  findings still needs a non-empty `results` list to be spec-valid, so it
  gets one synthetic passing `Result` recording "scan ran, nothing found"
  rather than being silently invalid or omitted.
- **`capability_map`/`policy_records` carry through as metadata, not a
  forced mapping.** Neither has a clean 1:1 shape in EvalPort's `Result`/
  `GraderResult` today (see the open question noted in the issue thread), so
  both ride along under `ResultSet.metadata.nuguard` — a tool-coverage
  summary and a policy-record count — rather than being dropped or
  shoehorned into fields that don't fit.

See `README.md` for install/usage and `src/nuguard_openeval_adapter/__init__.py`
for the full mapping with field-by-field rationale in the docstrings.

## User benefit

A nuguard `validate` run's output becomes exportable, once, to a format any
EvalPort-consuming tool already understands: aggregate nuguard results
alongside DeepEval/Promptfoo/CrewAI/etc. runs in one dashboard, diff a
`ResultSet` across runs with EvalPort's own tooling, or pipe nuguard
findings into any pipeline that already speaks EvalPort — without that
pipeline needing to know nuguard's `Finding` schema at all. This is
converter-only: it changes nothing about how nuguard runs scans or scores
findings, and nuguard's own reporting is untouched.

## Open questions (carried over from the issue thread, not resolved here)

- Whether `capability_map`/`policy_records` deserve a first-class EvalPort
  extension point of their own (rather than free-form `metadata`) is an
  EvalPort spec question, not something this adapter should decide
  unilaterally — flagging it for review rather than picking an answer.
- The severity→score banding (`critical`→0.0 ... `info`→1.0, five fixed
  steps) is one reasonable choice among several (e.g. using `ngrs_score`/100
  directly as a continuous score); this adapter uses the discrete banding by
  default with `ngrs_score` as a documented fallback only when `severity`
  doesn't parse — open to feedback if continuous NGRS is preferred.
