# EvalPort — Adoption Strategy (Updated September 27, 2026)

## Status: Phase 3 — One Docs Listing, Three Merged Upstream PRs (Two by the EvalPort Maintainer, One Bot-Generated from an EvalPort Issue), One Pending, 67 Adapters in This Repo

Every upstream status below was re-checked live against the GitHub API on 2026-09-27: PR `state`/`merged`/`merged_at`, issue state, and whether the merged files are present on the upstream default branch. The 2026-09-16 snapshot this replaces had drifted. It said Inspect AI filed EvalPort under `Tooling` (it is now under "Frameworks"), credited the IBM/ares approval to a reviewer who reviewed only the predecessor PR, and did not say who authored each merged change.

**Read this first:** every merged upstream change below started with the EvalPort maintainer (@adhabnr-ux), either as the PR author or as the person who filed the originating issue. They are real, merged, and present on each project's `main`, but none is an independent adoption by the upstream project.

### Listed in Inspect AI's Community Extensions (docs listing, not a code integration)

- **UKGovernmentBEIS/inspect_ai#4797** — "Add EvalPort to community extensions list," authored by @adhabnr-ux, merged 2026-08-11T16:30:26Z. It adds a six-line entry to `docs/extensions/extensions.yml`. On `main` today the entry is under `categories: ["Frameworks"]`; the PR proposed `Tooling`. The originating proposal (inspect_ai#4681, closed) asked for native `to_openeval()`/`from_openeval()` in Inspect AI; what landed is the docs listing only.

### Merged Upstream

- **IBM/ares#583** — `experimental-plugins/ares-openeval-adapter`, **authored by @adhabnr-ux**, merged 2026-09-04T12:33:39Z, `review_decision: APPROVED`, 22/22 checks green. GitHub search matches #583 for `reviewed-by:stefano81`, not for `reviewed-by:nedshivina`; @nedshivina reviewed the predecessor #579 (closed unmerged) and is still listed as a requested reviewer on #583. The plugin converts ARES attack goals and `AttackEval` output to/from EvalPort. It is an optional community plugin under ARES's `experimental-plugins/` directory, not an IBM-authored integration. Present on `main`: `experimental-plugins/ares-openeval-adapter/pyproject.toml`.
- **TIGER-AI-Lab/ClawBench#336** — `scripts/export_openeval.py`, **authored by @adhabnr-ux**, merged 2026-09-08T05:07:19Z. It resolves ClawBench#322 (filed by @adhabnr-ux, closed as completed). According to the PR description, it is a standalone script rather than a package because the maintainer asked for that. It is optional and adds no dependency to ClawBench. Present on `main`: `scripts/export_openeval.py`.
- **MervinPraison/PraisonAI#4667** — `to_evalport()`/`from_evalport()`/`report_to_evalport()` in `praisonaiagents.eval` (`src/praisonai-agents/praisonaiagents/eval/evalport.py`, present on `main`). **Opened by `praisonai-triage-agent[bot]`**, the project's automated triage/merge pipeline, in response to PraisonAI#4275, an issue filed by @adhabnr-ux. Merged 2026-09-02T11:38:09Z with no human review recorded (`review_decision: null`). **Caveat:** the merged module uses its own field names (`evalport_version`, `kind`, `cases`, `case_id`, `suite_name`), not the spec's (`version`, `id`, `test_cases`, `test_case_id`, `suite_id`, `run_id`, `started_at`). A suite and a ResultSet built exactly as that module builds them both fail `evalport-sdk`'s `validate_suite()`/`validate_result_set()` (checked in this pass). So this is a merged module named after EvalPort, but it does not yet emit spec-conformant EvalPort.

### Open, Not Merged

- **truera/trulens#2757** (TruLens, owned by Snowflake) — a `to_openeval()`/`from_openeval()` connector, **authored by @adhabnr-ux**. The PR description reports 17/17 of its tests passing against real `trulens-core`. Live state on 2026-09-27: `state: open`, `merged: false`, `mergeable: false` / `mergeable_state: dirty` (needs a rebase onto `main`), `review_decision: REVIEW_REQUIRED`, and 3 failing checks: `PR Validation Eval`, `PR Validation Eval (PRBranchProtect py311-static)`, and `cla`. The maintainer approval was given on its predecessor **truera/trulens#2697**, which was closed without merging on 2026-09-02 when its branch was recreated. #2757 has not been approved yet. The CLA has to be signed by the repository owner personally; the rebase and the two failing CI checks also have to be fixed. (An earlier `py310-static` failure was traced to a pre-existing upstream bug, filed as truera/trulens#2764 and closed as completed on 2026-09-10. The current `py311-static` failure has not been diagnosed and should not be assumed to be the same issue.)
- **harbor-framework/terminal-bench-science#1652** — `tools/tbscience-openeval-adapter`, **authored by @adhabnr-ux**, following a scope agreed in discussion #1600. Live state: `state: open`, `merged: false`, `review_decision: REVIEW_REQUIRED`. It has no approving review (search `review:approved` returns nothing); the maintainer has commented. Earlier README wording called it "approved," which was not accurate.
- **microsoft/autogen#8009** — an OpenEval adapter opened by an independent community contributor (@DresdenGman, not affiliated with this project). Still open and still a draft, `review_decision: REVIEW_REQUIRED`, last updated 2026-08-22. The issue it addresses, autogen#8005, is also still open. Nothing actionable from our side.

### Proposed and Closed Without Merging

- **openai/openai-python#3619** — OpenEval import/export helpers written by an independent contributor (@SparshGarg999). **Closed without merging on 2026-08-25** (`merged: false`, `merged_at: null`). The originating proposal, openai-python#3549 (filed by @adhabnr-ux), was closed as "not planned" the same day. CONTRIBUTORS.md used to describe this as built "directly into" openai-python; it never shipped.
- **deepsense-ai/ragbits#989** — an `EvalPortDataLoader` written by ragbits maintainer @mikemikimike after a proposal in ragbits#986 (filed by @adhabnr-ux, still open). **Closed without merging on 2026-09-10.** `docs/community-integrations.md` used to list it as open and pending review.

### Framework Adapters in This Repo: 67

Installable packages under `adapters/` in this repo (67 directories on `main` as of 2026-09-29; 65 on 2026-09-27, up from 54 on 2026-09-16). Each has a `pyproject.toml` depending on `evalport-sdk`, `to_openeval()`/`from_openeval()`, and its own test suite, which CI's adapter job runs against the real validator. 64 were written by the EvalPort maintainer and 3 by external contributors (see CONTRIBUTORS.md). They are converters maintained by this project, built against each framework's public data shapes. They are not integrations shipped by, or adopted by, those frameworks.

The two added since the 2026-09-27 snapshot, both built at the upstream maintainer's explicit request rather than unsolicited: [`agent-skills-eval-openeval-adapter`](adapters/agent-skills-eval-openeval-adapter/), following [darkrishabh/agent-skills-eval#34](https://github.com/darkrishabh/agent-skills-eval/issues/34#issuecomment-5888398803) (maintainer @darkrishabh: "A standalone adapter in EvalPort is the direction we'd prefer... No first-party export or dependency is planned at this stage"); and [`humanbound-openeval-adapter`](adapters/humanbound-openeval-adapter/), following [humanbound/humanbound#131](https://github.com/humanbound/humanbound/issues/131#issuecomment-5885427531) (maintainer @sotberd: "We'd prefer the standalone package in the EvalPort repo, as you suggested... Once it's published, share the link here and we'll be happy to review it for a mention in our docs"). Neither of these two claims has been independently re-verified against a fresh live GitHub check the way the rest of this document's 2026-09-27 snapshot was — both are read directly from the linked comments, not from an API re-check performed on 2026-09-29.

### Published Packages

Re-checked live against the registries on 2026-09-27 (unchanged since 2026-09-16):

| Package | Registry | Live version |
|---------|----------|---------------|
| evalport-sdk | PyPI | 1.3.1 |
| evalport-sdk | npm | 1.3.1 |
| evalport-cli | npm | 1.0.0 |

`evalport-sdk` on both registries now matches this repo's current SDK version, resolving the 1.0.0-vs-1.1.0 mismatch the last snapshot of this document flagged. `evalport-cli` on npm is still at 1.0.0 and has not been re-published alongside the SDK's later releases — noting this rather than letting it sit unflagged.

### Collaborators

As of 2026-09-16 (not re-checked in the 2026-09-27 pass): [SparshGarg999](https://github.com/SparshGarg999) holds `write` access (accepted, unchanged since 2026-08-16). The [DresdenGman](https://github.com/DresdenGman) invitation referenced in the last snapshot is no longer showing as pending in a live collaborator check — recorded here as a state change rather than silently dropped; the reason for the change (declined, expired, or something else) hasn't been independently confirmed.

### Outreach Volume

A live GitHub search for issues authored by this project mentioning "evalport" or "openeval" (`author:adhabnr-ux is:issue evalport OR openeval in:title,body`) returned **691** results on 2026-09-16 (not re-run in the 2026-09-27 pass), up from 151 at the snapshot before that. This number is an activity count, not a success metric — most of these are individual outreach threads across many repositories, the large majority of which end in a decline, a "not now," or no reply at all, which is the normal shape of unsolicited technical outreach. The merged changes and the open TruLens PR above are the actual signal, with the authorship caveats stated there; this count is included only because the last version of this document tracked it and dropping a previously-tracked number silently would be worse than keeping it with this caveat attached.

### Social Media (unchanged since August 16; not re-verified this pass)
- Hacker News: https://news.ycombinator.com/item?id=49105771
- Dev.to: https://dev.to/adha_ak_d60b39fbb66769fd1/openeval-why-llm-evaluation-needs-a-standard-format-50di
- LinkedIn: https://www.linkedin.com/feed/update/urn:li:share:7488433286059347969/
- Reddit: blocked (karma requirements) as of last check

### Next Steps
1. Rebase truera/trulens#2757 onto current `main`, then diagnose the `PR Validation Eval` and `py311-static` failures (don't assume it's the same issue as the now-fixed `py310-static`/#2764 bug). Sign the TruLens CLA (repository owner only — this is a personal legal attestation of authorship, not something a session working on the owner's behalf can post). Both are needed before this can merge.
2. Once truera/trulens#2757 merges, promote the spec from release-candidate to 1.0.0 final, per `CRITIQUE.md`'s own stated release criterion.
3. Republish `evalport-cli` on npm so it matches the SDK's 1.3.1 line.
4. No action pending on microsoft/autogen#8009; watch for maintainer review activity rather than re-pinging a contributor who isn't us.
5. Re-run the `author:adhabnr-ux is:issue evalport OR openeval` search periodically and continue reading every reply in full before responding, per the standing "never fabricate a maintainer's stance" rule.
6. Decide whether to offer PraisonAI a follow-up PR that makes `praisonaiagents.eval.evalport` emit spec-valid Suites/ResultSets. It does not today (see the caveat above). Until it does, do not describe that module as producing EvalPort output.
