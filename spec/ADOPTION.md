# EvalPort — Adoption Strategy (Updated September 16, 2026)

## Status: Phase 3 — Four Merged Third-Party Integrations, One Pending on a CLA, 54 Shipped Adapters

Every claim below was re-verified live against GitHub and the package registries on 2026-09-16 while refreshing this document (not carried over from the August 16 snapshot, which had gone stale on several counts by the time this pass started).

### Merged Integrations

- **Merged: IBM/ares#583** — `experimental-plugins/ares-openeval-adapter`, merged 2026-09-04, reviewed and approved by @nedshivina, all 22 CI checks green. Converts ARES attack goals and `AttackEval` output to/from EvalPort. This is EvalPort's first shipped integration into a named enterprise vendor's own repository (IBM's AI Robustness Evaluation System).
- **Merged: MervinPraison/PraisonAI#4667** — native `to_evalport()`/`from_evalport()`/`report_to_evalport()` added directly to `praisonaiagents.eval` (the core package, not a separate adapter package), merged 2026-09-02 via the maintainers' own automated triage/merge pipeline. Zero new dependency on the core SDK.
- **Merged: TIGER-AI-Lab/ClawBench#336** — `scripts/export_openeval.py`, merged 2026-09-08 after the maintainer (@Perry2004) specified where it should live (a script, not a package). ClawBench is a 750+-star open benchmark for browser AI agents.
- **Merged: UKGovernmentBEIS/inspect_ai#4797** — "Add EvalPort to community extensions list," merged 2026-08-11 (unchanged since last update). EvalPort is listed in Inspect AI's official community extensions under `Tooling`.

### Pending on a Human Action, Not on Us

- **truera/trulens#2757** (TruLens, owned by Snowflake) — a full `to_openeval()`/`from_openeval()` connector, 18/18 tests passing against real `trulens-core`, all CI green except one pre-existing Python 3.10 bug unrelated to this PR (filed separately and independently reproduced as truera/trulens#2764, confirmed on real Python 3.10.20). The only remaining blocker is TruLens's CLA bot, which requires the repository owner to personally post a signed attestation on the PR — not something that can be done on their behalf. Everything on the engineering side is finished and re-verified as of 2026-09-15.

### Not Yet Resolved

- **microsoft/autogen#8009** — an OpenEval adapter opened by an independent community contributor (@DresdenGman, not affiliated with this project), still open and in draft, `review_decision: REVIEW_REQUIRED`, no maintainer activity since 2026-08-22. Nothing actionable from our side; noting it here rather than dropping it from tracking.

### Shipped Framework Adapters: 54

Real, installable packages under `adapters/<name>-openeval-adapter/` in this repo (pyproject.toml depending on `evalport-sdk`, `to_openeval()`/`from_openeval()`, tests run against the real validator, README) — verified by listing the directory live on 2026-09-16, not carried over from an earlier count. This is up from 20 at the last snapshot (2026-08-16).

### Published Packages

Verified live against the registries on 2026-09-16 (not assumed):

| Package | Registry | Live version |
|---------|----------|---------------|
| evalport-sdk | PyPI | 1.3.1 |
| evalport-sdk | npm | 1.3.1 |
| evalport-cli | npm | 1.0.0 |

`evalport-sdk` on both registries now matches this repo's current SDK version, resolving the 1.0.0-vs-1.1.0 mismatch the last snapshot of this document flagged. `evalport-cli` on npm is still at 1.0.0 and has not been re-published alongside the SDK's later releases — noting this rather than letting it sit unflagged.

### Collaborators

Live as of 2026-09-16: [SparshGarg999](https://github.com/SparshGarg999) holds `write` access (accepted, unchanged since 2026-08-16). The [DresdenGman](https://github.com/DresdenGman) invitation referenced in the last snapshot is no longer showing as pending in a live collaborator check — recorded here as a state change rather than silently dropped; the reason for the change (declined, expired, or something else) hasn't been independently confirmed.

### Outreach Volume

A live GitHub search for issues authored by this project mentioning "evalport" or "openeval" (`author:adhabnr-ux is:issue evalport OR openeval in:title,body`) returns **691** results as of this refresh, up from 151 at the last snapshot. This number is an activity count, not a success metric — most of these are individual outreach threads across many repositories, the large majority of which end in a decline, a "not now," or no reply at all, which is the normal shape of unsolicited technical outreach. The four merged integrations and the one CLA-pending integration above are the actual signal; this count is included only because the last version of this document tracked it and dropping a previously-tracked number silently would be worse than keeping it with this caveat attached.

### Social Media (unchanged since August 16; not re-verified this pass)
- Hacker News: https://news.ycombinator.com/item?id=49105771
- Dev.to: https://dev.to/adha_ak_d60b39fbb66769fd1/openeval-why-llm-evaluation-needs-a-standard-format-50di
- LinkedIn: https://www.linkedin.com/feed/update/urn:li:share:7488433286059347969/
- Reddit: blocked (karma requirements) as of last check

### Next Steps
1. Sign the TruLens CLA on truera/trulens#2757 (repository owner only — this is a personal legal attestation of authorship, not something a session working on the owner's behalf can post). Once signed, the PR is otherwise ready to merge.
2. Once truera/trulens#2757 merges, promote the spec from release-candidate to 1.0.0 final, per `CRITIQUE.md`'s own stated release criterion.
3. Republish `evalport-cli` on npm so it matches the SDK's 1.3.1 line.
4. No action pending on microsoft/autogen#8009; watch for maintainer review activity rather than re-pinging a contributor who isn't us.
5. Re-run the `author:adhabnr-ux is:issue evalport OR openeval` search periodically and continue reading every reply in full before responding, per the standing "never fabricate a maintainer's stance" rule.
