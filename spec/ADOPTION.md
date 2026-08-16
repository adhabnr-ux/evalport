# EvalPort — Adoption Strategy (Updated August 16, 2026)

## Status: Phase 2 — First Merge, One Accepted Collaborator, 20 Shipped Adapters, Governance + RFC Process Formalized

Every claim below was re-verified live against GitHub/PyPI/npm on 2026-08-16 while refreshing this document (not carried over from an earlier, unverified snapshot).

### Key Milestones

- **Merged: UKGovernmentBEIS/inspect_ai#4797** — "Add EvalPort to community extensions list," merged 2026-08-11. EvalPort is now listed in Inspect AI's official community extensions under the `Tooling` category. This is EvalPort's first shipped integration into a third-party framework's own repository.
- **In active review: truera/trulens#2697** — a full `to_openeval()`/`from_openeval()` module for TruLens, closing truera/trulens#2680. Open, `review_decision: REVIEW_REQUIRED`. Two real bugs surfaced by early CI runs were found and fixed on this PR (a missing `trulens-openeval`/`evalport-sdk` entry in `pyproject.toml`'s required dependency group, and the corresponding `poetry.lock` update) — `tests/unit/test_openeval.py` now passes 17/17 on every job that reaches it. The 3 checks still showing red (`PR Validation Eval` and its py310/py313 static variants) were dug into via the Azure Pipelines logs directly and root-caused to two pre-existing issues unrelated to this PR: flaky timing in `test_otel_async_concurrency.py`, and a segfault during collection of the heavy ML dependency stack in the optional/snowflake suites on py310/py313 only. Diagnosis posted to the PR; nothing further actionable from the adapter side, waiting on maintainer review.
- **Open, draft: microsoft/autogen#8009** — the original first integration PR (DresdenGman, `autogenstudio/eval/openeval.py`). Still open, still in draft, `review_decision: REVIEW_REQUIRED`, 2/2 checks passing. No maintainer merge decision yet.
- **Governance formalized, spec bumped to 1.0.0-rc.2 (2026-08-16):** `spec/SPEC.md` now has a `## Governance` section documenting current stewardship, the RFC process (restated in full inside the spec itself, not just `.github/CONTRIBUTING.md`), and the actual path to becoming a collaborator. A new `## Open Design Questions` table links every item `spec/CRITIQUE.md` flags as deliberately deferred to a live GitHub Discussion where the design work happens: [#8 suite/result signing](https://github.com/adhabnr-ux/evalport/discussions/8), [#9 formal conformance test suite](https://github.com/adhabnr-ux/evalport/discussions/9), [#10 resumable runs / partial ResultSet merging](https://github.com/adhabnr-ux/evalport/discussions/10), [#11 llm_judge injection mitigations, MUST vs SHOULD](https://github.com/adhabnr-ux/evalport/discussions/11). No comments on any of the four yet (all opened 2026-08-16). This is aimed directly at outside contributors: someone who wants to shape the spec (not just ship a framework adapter) now has four concrete, scoped entry points with no prior EvalPort contribution required.
- **First accepted collaborator (2026-08-16):** [SparshGarg999](https://github.com/SparshGarg999) accepted the collaborator invitation sent after shipping the openai-python OpenEval adapter (openai/openai-python#3619) — confirmed via a live `get_repo_collaborators` call, which now lists them with `write` access, not just a pending invite. [DresdenGman](https://github.com/DresdenGman) (AutoGen #8009) was invited the same day and is still "Pending Invite" as of this refresh. `CONTRIBUTORS.md` credits both, plus the two maintainers who've substantively engaged (Josh Reini / TruLens, Charles Teague / TruLens).

### Shipped Framework Adapters: 20

Real, installable Python packages under `adapters/<name>-openeval-adapter/` (pyproject.toml depending on `evalport-sdk`, `to_openeval()`/`from_openeval()`, tests run against the real validator, README): Argilla, AutoGen, Braintrust, CrewAI, DSPy, Evidently, Giskard, Guardrails, Haystack, Langfuse, LangSmith, LlamaIndex, MLflow, Opik (Comet), Patronus, Phoenix (Arize), Ragas, uptrain, Vertex AI (`vertexai.evaluation`), Weave (Weights & Biases).

### Published Packages

Verified live on 2026-08-16 (registry lookups, not assumed):

| Package | Registry | Live version | URL |
|---------|----------|---------------|-----|
| evalport-sdk | npm | 1.0.0 | https://www.npmjs.com/package/evalport-sdk |
| evalport-cli | npm | — | https://www.npmjs.com/package/evalport-cli |
| evalport-sdk | PyPI | 1.0.0 | https://pypi.org/project/evalport-sdk/ |

The repository's own SDK sources are now at package version **1.1.0** (`sdk/python/pyproject.toml`, `sdk/typescript/package.json` — see their Change Log/comments for what changed) and spec version **1.0.0-rc.2** (`spec/SPEC.md`). Publishing 1.1.0 to PyPI/npm requires a maintainer to run `twine upload`/`npm publish` with credentials this repository's automation does not have (see `MANUAL-ACTIONS.md`) — until that manual step happens, installing `pip install evalport-sdk` / `npm install evalport-sdk` still gets 1.0.0, which does not yet include the semver/grader-type/score-range validator fixes documented in `spec/SPEC.md`'s Change Log.

### Outreach Volume

A live GitHub search for issues authored by this project mentioning "evalport" or "openeval" returns **151 results** as of this refresh (`author:adhabnr-ux is:issue evalport OR openeval in:title,body`) — this count includes prior outreach rounds beyond what earlier snapshots of this document tracked, and was not manually re-triaged repo-by-repo in this pass. The per-repo "Active Conversations" table from the previous version of this document is retired in favor of this live, re-runnable query, since a manually-maintained snapshot list goes stale (as this document itself demonstrably did between July 30 and August 16).

### Social Media (unchanged since last update; not re-verified this pass)
- Hacker News: https://news.ycombinator.com/item?id=49105771
- Dev.to: https://dev.to/adha_ak_d60b39fbb66769fd1/openeval-why-llm-evaluation-needs-a-standard-format-50di
- LinkedIn: https://www.linkedin.com/feed/update/urn:li:share:7488433286059347969/
- Reddit: blocked (karma requirements) as of last check

### Next Steps
1. Get a maintainer merge decision on truera/trulens#2697 — the adapter itself is done and tested (17/17), and the CI-red diagnosis has been posted; nothing further to do until a maintainer reviews.
2. Get a maintainer merge decision on microsoft/autogen#8009 (open since 2026-07-30, still unreviewed as of 2026-08-16).
3. When credentials are available, publish `evalport-sdk` 1.1.0 to PyPI and npm so `pip install`/`npm install` pick up the semver, grader-type, and score-range validator fixes.
4. Re-run the `author:adhabnr-ux is:issue evalport OR openeval` search periodically and triage new maintainer replies substantively, per the standing "read every reply, never fabricate a maintainer's stance" rule.
5. Watch Discussions #8-#11 for replies and follow up substantively; check whether DresdenGman accepts their pending collaborator invitation.
