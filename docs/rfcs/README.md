# EvalPort RFCs

Every change to the EvalPort spec starts as an RFC: a GitHub Discussion in the
[Ideas](https://github.com/adhabnr-ux/evalport/discussions/categories/ideas) category.
This page lists every RFC the project has run and where each one stands.
The process itself is defined in [`spec/SPEC.md` → Governance](../../spec/SPEC.md#governance).

**Five RFCs are open for comment right now.** You don't need to have contributed
before to comment on one.

Two of them, #47 and #49, touch what a `Result` says about its own outcome. [`outcome-model.md`](outcome-model.md) puts them side by side: five situations, how each is written on `main` today, and what each proposal would add.

## Index

Dates are the PR merge date (UTC) where a PR landed the change. Otherwise they are the
date in the [`spec/SPEC.md` Change Log](../../spec/SPEC.md#change-log).

### Open

| RFC | Question | Status | Discussion | Reference PR |
|---|---|---|---|---|
| `Result.constraint_violations` | Should a hard constraint (e.g. an RBAC or governance violation) be a first-class part of a `Result`? It would fail the result outright but, unlike an `error` row, keep it in denominators, and it would not be averaged in like a grader score. | **Waiting on a real fixture.** The two-week comment period ran through 2026-09-23 and was deliberately left open. The RFC is waiting on a conformance fixture from a real AOBench RBAC hard-fail run, offered by AOBench's maintainer (@MSKazemi), whose framing the proposal is built on. | [#47](https://github.com/adhabnr-ux/evalport/discussions/47) | [#48](https://github.com/adhabnr-ux/evalport/pull/48) (draft, not for merge) |
| `Result.verdict`: FAILED vs. UNVERIFIED | Should a `Result` be able to say "this reached a terminal state nobody could judge", as distinct from a graded failure (`passed: false`) and a harness failure (`error`)? Raised by @soul-sol. | **Open. Comment period 2026-09-27 to 2026-10-11.** | [#49](https://github.com/adhabnr-ux/evalport/discussions/49) | [#83](https://github.com/adhabnr-ux/evalport/pull/83) (draft, not for merge) |
| `GraderResult.trials` / `successes` | Should a rate-based grader score carry its denominator, so 0 of 5 and 0 of 500 don't both serialize as `0.0`? And should it be optional, required for rate-based graders, or a `metadata` convention? Promoted from [issue #58](https://github.com/adhabnr-ux/evalport/issues/58), raised by @sattyamjjain. | **Open. Comment period 2026-09-27 to 2026-10-11.** | [#67](https://github.com/adhabnr-ux/evalport/discussions/67) | [#66](https://github.com/adhabnr-ux/evalport/pull/66) (draft, implements the optional-fields option only, not for merge) |
| Unknown fields: strict validation, lenient consumption | The schemas close 16 objects with `additionalProperties: false`, the SDK validators accept unknown keys, and the spec says runners "MUST ignore unknown fields". Should validation be strict by default (new code `UNKNOWN_FIELD`) with an explicit opt-in for lenient consumption? Promoted from [issue #107](https://github.com/adhabnr-ux/evalport/issues/107), found by @ChelseaKR. | **Open. Comment period 2026-09-29 to 2026-10-13.** | [#108](https://github.com/adhabnr-ux/evalport/discussions/108) | [#110](https://github.com/adhabnr-ux/evalport/pull/110) (draft, not for merge) |
| Judge identity, declared vs. observed | A suite declares the judge it wants (`llm_judge` `params.model`, `prompt`), but a `GraderResult` has no typed place for the judge that actually answered. Should EvalPort define an unvalidated convention, `metadata.openeval.judge`, for a runner to self-report it, rather than a typed field? Raised by @jonathanngiroux-star (Proofspan) and @joslat (AgentEval). | **Open. Comment period 2026-10-03 to 2026-10-17.** | [#118](https://github.com/adhabnr-ux/evalport/discussions/118) | [#119](https://github.com/adhabnr-ux/evalport/pull/119) (draft, convention text and one fixture only, not for merge) |

### Landed

| RFC | Question | Status | Discussion | Reference PR |
|---|---|---|---|---|
| `ResultSet.group` | How should sibling `ResultSet`s (mutation-testing runs, seed sweeps, model comparisons) be joined into a group? | **Landed in [PR #54](https://github.com/adhabnr-ux/evalport/pull/54) on 2026-09-20**, after the comment period closed on 2026-09-18. Unreleased (1.0.0-rc.6). | [#45](https://github.com/adhabnr-ux/evalport/discussions/45) | [#46](https://github.com/adhabnr-ux/evalport/pull/46) (closed, replaced by #54) |
| `Result.attempt` / `ResultSet.isolation` | How should a `ResultSet` represent repeated attempts of the same test case (`num_repetitions`, epochs)? Raised by AgentVerity's maintainer in [issue #20](https://github.com/adhabnr-ux/evalport/issues/20). | **Landed in [PR #35](https://github.com/adhabnr-ux/evalport/pull/35) on 2026-09-01**, released as 1.0.0-rc.5. | [#22](https://github.com/adhabnr-ux/evalport/discussions/22) | none (#35 was the implementation) |
| Adapter packaging convention | Should each adapter declare its target framework as a pinned optional extra, not only in `test`? A packaging convention, not a change to the data format. | **Resolved on 2026-08-21** in `.github/CONTRIBUTING.md`'s adapter checklist (commit `bb3bcae`, no PR). | [#13](https://github.com/adhabnr-ux/evalport/discussions/13) | none |
| Suite/ResultSet signing | How can a consumer verify that a published suite or result set wasn't modified after publication? | **Landed on 2026-08-22** in 1.0.0-rc.4: detached Sigstore signatures and `spec/tools/verify_signature.py`. Committed directly to `main`, no PR. | [#8](https://github.com/adhabnr-ux/evalport/discussions/8) | none |
| Conformance test suite | Should there be a spec-owned set of fixtures that any implementation, in any language, can check itself against? | **Landed on 2026-08-22** in 1.0.0-rc.3: `spec/conformance/`. Committed directly to `main`, no PR. | [#9](https://github.com/adhabnr-ux/evalport/discussions/9) | none |
| Resumable runs and partial `ResultSet`s | How should a partial `ResultSet` be marked, and how should two partial ones be merged? | **Landed on 2026-08-22** in 1.0.0-rc.3: `Result.completed_at` and `metadata.openeval.partial`. Committed directly to `main`, no PR. | [#10](https://github.com/adhabnr-ux/evalport/discussions/10) | none |
| `llm_judge` injection mitigations | Should `llm_judge` prompt-injection mitigations be a MUST instead of a SHOULD? | **Landed on 2026-08-22** in 1.0.0-rc.3. The mitigations stay SHOULDs; runners can self-report what they applied via `metadata.openeval.judge_hardening`. Committed directly to `main`, no PR. | [#11](https://github.com/adhabnr-ux/evalport/discussions/11) | none |

A landed RFC's Discussion stays open. Several of them still have follow-up questions,
and pushback on a shipped design is welcome there.

## How to take part

Comment on the Discussion. That's the whole mechanism: no sign-up, and no prior
contribution needed. The Governance section says it plainly: "a well-reasoned objection
from a first-time contributor carries the same weight as one from a collaborator."

Comments that move an RFC forward:

- **A counter-example from a real system.** For example, an eval harness, runtime, or
  benchmark whose output doesn't fit the proposed shape, or fits it only by losing
  information. Link to the code or docs if you can.
- **An objection to a field name or a rule.** Names are cheap to change during the
  comment period and expensive once adapters depend on them. If a name misleads, or a
  validation rule rejects something legitimate, say so.
- **A fixture from real output.** A document produced by a real run that the proposal
  should accept or reject. Each RFC's draft reference PR adds fixtures under
  `spec/conformance/fixtures/`, so you can check yours against the branch.
- **"This already exists elsewhere."** Prior art from another format that solved (or badly
  solved) the same problem.

Each open RFC's Discussion ends with specific questions for commenters. Those are a good
place to start.

## How an RFC moves

This summarizes [`spec/SPEC.md` → Governance](../../spec/SPEC.md#governance), which is
authoritative:

1. **Proposal.** A Discussion in the Ideas category states the problem, the proposed
   change, and its impact on backward compatibility.
2. **Two-week comment period.** A date is a floor, not a trigger: #47 is waiting on a real
   fixture past its original end date.
3. **Rough consensus, then a PR.** The change is implemented in one PR covering
   `spec/SPEC.md` (mirrored to the root `SPEC.md`), the JSON Schemas, and both reference
   SDKs. In Governance's words, "a spec change that doesn't touch the SDKs' validators
   isn't actually specified, it's aspirational."
4. **Breaking changes** also need the spec lead's sign-off, whatever the comment-period
   consensus.

In practice, recent RFCs have also had a **draft reference-implementation PR** open during the comment
period (#46, #48, #66, #83, #110). It is marked as not for merge and exists so reviewers can test
a concrete design. After consensus, the implementation is re-opened fresh against `main`,
as #54 did for #46.

To propose a new RFC, open a Discussion in
[Ideas](https://github.com/adhabnr-ux/evalport/discussions/categories/ideas) titled
`[Spec Change] <short description>`.
