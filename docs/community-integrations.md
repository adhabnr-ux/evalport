# Community Integrations

This page tracks EvalPort support that other projects have written in their own
codebases: either merged, or proposed in an upstream PR. Each entry says whether
the work merged, who wrote it, and whether its output actually validates.
Statuses were checked live against GitHub on **2026-09-30**.

These are not the same as [`adapters/`](../adapters/). Those packages are
`to_openeval()`/`from_openeval()` converters that EvalPort itself writes and
maintains. The entries below are owned by the other project.

**Provenance.** Every entry below followed an issue or proposal filed by the
EvalPort maintainer (@adhabnr-ux). None of them arrived unprompted. The line
that matters is who wrote and merged the code: the upstream project, a bot, or
an outside contributor.

## Merged upstream

### gauntlet (ChelseaKR/gauntlet)

- **What it adds:** `gauntlet report --format evalport`, written by the
  project's owner.
- **Merged:** [ChelseaKR/gauntlet#76](https://github.com/ChelseaKR/gauntlet/pull/76)
  on 2026-09-19. It followed
  [gauntlet#27](https://github.com/ChelseaKR/gauntlet/issues/27). It is on
  `main` but not yet in a tagged release; the latest tag is v0.3.0.
- **Withheld runs are not exported.** A run whose verdict gauntlet withholds as
  unscoreable is not written as a ResultSet at all: the command exits 4 instead.
  EvalPort currently has no way to say "this run has no verdict", so this is
  the right behaviour.
- **Spec feedback it produced:**
  - The reference SDK accepted documents the published schema rejects. The
    type-checking half was fixed in #106. The unknown-field half is a spec
    question, tracked in #107 and [Discussion #108](https://github.com/adhabnr-ux/evalport/discussions/108).
  - There is no suite-level threshold.
  - A multi-turn case has only one `actual_output` string.
  - There is no run-level "verdict withheld" state.

### PraisonAI (MervinPraison/PraisonAI)

- **What it adds:** `to_evalport()`, `from_evalport()` and `report_to_evalport()`
  in `praisonaiagents.eval`.
- **Merged:** [MervinPraison/PraisonAI#4667](https://github.com/MervinPraison/PraisonAI/pull/4667)
  on 2026-09-02. The PR was opened by the project's automated triage bot in
  response to [PraisonAI#4275](https://github.com/MervinPraison/PraisonAI/issues/4275),
  and no human review is recorded.
- **Its output does not currently validate.** The module uses its own field
  names (`cases`, `case_id`, `evalport_version`, ...). As a result:
  - `validate_suite()` fails.
  - `validate_result_set()` fails.
  - The raw JSON Schemas also reject it.
- **Fix proposed:** a tested, dependency-free patch is in
  [PraisonAI#5374](https://github.com/MervinPraison/PraisonAI/issues/5374).
  Until that lands, do not describe this module as producing EvalPort output.

## Open upstream PRs

### meta-llama/llama-cookbook#1080

- **PR:** [meta-llama/llama-cookbook#1080](https://github.com/meta-llama/llama-cookbook/pull/1080)
  adds `to_evalport.py`, which exports the synthetic-evaluation notebook's
  generated context/report pairs as an EvalPort suite. It was written by an
  outside contributor and follows
  [llama-cookbook#1072](https://github.com/meta-llama/llama-cookbook/issues/1072).
- **Status:** open and awaiting maintainer review. CI is green.
- **Checked on the PR head:**
  - The exported suite passes evalport-sdk 1.3.1 and the raw schema.
  - Its 13 tests pass.
- **Correction on our side:** the PR also caught an invalid grader sketch in
  our own #1072 issue.

### JudgmentLabs/judgeval#786

- **PR:** [JudgmentLabs/judgeval#786](https://github.com/JudgmentLabs/judgeval/pull/786)
  adds an EvalPort ResultSet adapter for Judgeval `ScoringResult`s, with no
  runtime dependency. It was written by an outside contributor and follows
  [judgeval#778](https://github.com/JudgmentLabs/judgeval/issues/778).
- **Status:** open and awaiting maintainer review.
- **Our review found spec issues:**
  - A null score is reported with `passed: true`.
  - Grader-level `passed` copies the row verdict instead of the grader's own
    result.
  - Timestamps without an offset are accepted.
  - An errored numeric scorer aborts the whole export.

## Closed or unmergeable

### SolaceLabs/solace-agent-mesh#1653: repository archived

- **What it was:** an EvalPort bridge for SAM's `evaluation/` framework, with
  42 tests. It used `attempt` for SAM's run number. It was written by an outside
  contributor after
  [solace-agent-mesh#1635](https://github.com/SolaceLabs/solace-agent-mesh/issues/1635).
- **Why it can't merge:** the PR is still marked open, but the upstream
  repository is now **archived (read-only)**.
- **Validation:** its suite and ResultSet output validated against the SDK and
  the raw schemas.

### deepsense-ai/ragbits#989: closed without merging

- **What it was:** ragbits maintainer @mikemikimike wrote an
  `EvalPortDataLoader`, a `QuestionAnswerDataLoader` subclass defaulting to
  EvalPort's `input` / `expected_output` / `retrieval_context` field names. It
  followed [ragbits#986](https://github.com/deepsense-ai/ragbits/issues/986).
- **Closed:** [deepsense-ai/ragbits#989](https://github.com/deepsense-ai/ragbits/pull/989)
  was closed without merging on 2026-09-10, so the loader is not part of
  ragbits.

---

Know of another project with an EvalPort integration, merged or in an open
upstream PR? Open a PR adding it here, or start a
[Discussion](https://github.com/adhabnr-ux/evalport/discussions). We list
maintainer-owned projects only with their permission.
