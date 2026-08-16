# EvalPort — Adoption Strategy (Updated August 16, 2026)

## Status: Phase 2 — First Merge, 20 Shipped Adapters, One PR in Active Review

Every claim below was re-verified live against GitHub/PyPI/npm on 2026-08-16 while refreshing this document (not carried over from an earlier, unverified snapshot).

### Key Milestones

- **Merged: UKGovernmentBEIS/inspect_ai#4797** — "Add EvalPort to community extensions list," merged 2026-08-11. EvalPort is now listed in Inspect AI's official community extensions under the `Tooling` category. This is EvalPort's first shipped integration into a third-party framework's own repository.
- **In active review: truera/trulens#2697** — a full `to_openeval()`/`from_openeval()` module for TruLens, closing truera/trulens#2680. Open, `review_decision: REVIEW_REQUIRED`. As of this refresh, 3 of 9 CI checks are failing (`PR Validation Eval` and its py310/py313 static variants) — these need to be fixed before this can merge; not yet actioned in this pass.
- **Open, draft: microsoft/autogen#8009** — the original first integration PR (DresdenGman, `autogenstudio/eval/openeval.py`). Still open, still in draft, `review_decision: REVIEW_REQUIRED`, 2/2 checks passing. No maintainer merge decision yet.

### Shipped Framework Adapters: 20

Real, installable Python packages under `adapters/<name>-openeval-adapter/` (pyproject.toml depending on `evalport-sdk`, `to_openeval()`/`from_openeval()`, tests run against the real validator, README): Argilla, AutoGen, Braintrust, CrewAI, DSPy, Evidently, Giskard, Guardrails, Haystack, Langfuse, LangSmith, LlamaIndex, MLflow, Opik (Comet), Patronus, Phoenix (Arize), Ragas, uptrain, Vertex AI (`vertexai.evaluation`), Weave (Weights & Biases).

### Published Packages

Verified live on 2026-08-16 (registry lookups, not assumed):

| Package | Registry | Live version | URL |
|---------|----------|---------------|-----|
| evalport-sdk | npm | 1.0.0 | https://www.npmjs.com/package/evalport-sdk |
| evalport-cli | npm | — | https://www.npmjs.com/package/evalport-cli |
| evalport-sdk | PyPI | 1.0.0 | https://pypi.org/project/evalport-sdk/ |

The repository's own SDK sources are now at package version **1.1.0** (`sdk/python/pyproject.toml`, `sdk/typescript/package.json` — see their Change Log/comments for what changed) and spec version **1.0.0-rc.1** (`spec/SPEC.md`). Publishing 1.1.0 to PyPI/npm requires a maintainer to run `twine upload`/`npm publish` with credentials this repository's automation does not have (see `MANUAL-ACTIONS.md`) — until that manual step happens, installing `pip install evalport-sdk` / `npm install evalport-sdk` still gets 1.0.0, which does not yet include the semver/grader-type/score-range validator fixes documented in `spec/SPEC.md`'s Change Log.

### Outreach Volume

A live GitHub search for issues authored by this project mentioning "evalport" or "openeval" returns **151 results** as of this refresh (`author:adhabnr-ux is:issue evalport OR openeval in:title,body`) — this count includes prior outreach rounds beyond what earlier snapshots of this document tracked, and was not manually re-triaged repo-by-repo in this pass. The per-repo "Active Conversations" table from the previous version of this document is retired in favor of this live, re-runnable query, since a manually-maintained snapshot list goes stale (as this document itself demonstrably did between July 30 and August 16).

### Social Media (unchanged since last update; not re-verified this pass)
- Hacker News: https://news.ycombinator.com/item?id=49105771
- Dev.to: https://dev.to/adha_ak_d60b39fbb66769fd1/openeval-why-llm-evaluation-needs-a-standard-format-50di
- LinkedIn: https://www.linkedin.com/feed/update/urn:li:share:7488433286059347969/
- Reddit: blocked (karma requirements) as of last check

### Next Steps
1. Fix the 3 failing CI checks on truera/trulens#2697 so it's mergeable (not yet actioned — flagged here for a future pass, since this refresh was scoped to the spec/schema alignment work rather than PR maintenance).
2. Get a maintainer merge decision on microsoft/autogen#8009 (open since 2026-07-30, still unreviewed as of 2026-08-16).
3. When credentials are available, publish `evalport-sdk` 1.1.0 to PyPI and npm so `pip install`/`npm install` pick up the semver, grader-type, and score-range validator fixes.
4. Re-run the `author:adhabnr-ux is:issue evalport OR openeval` search periodically and triage new maintainer replies substantively, per the standing "read every reply, never fabricate a maintainer's stance" rule.
