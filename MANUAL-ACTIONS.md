# EvalPort — Manual Actions Checklist (Updated August 16, 2026)

This file previously listed launch-day tasks (npm/PyPI publish, founder emails, social posts) that are now either done or superseded. Rewritten to reflect what's actually outstanding as of this update — re-verified live, not carried over from the old version.

## What's already done (no action needed)

- GitHub repo live, spec at 1.0.0-rc.2, Governance section + 4 open RFC Discussions (#8–#11)
- 20 real, tested framework adapters shipped under `adapters/`
- Merged into UK AISI's Inspect AI community extensions list ([PR #4797](https://github.com/UKGovernmentBEIS/inspect_ai/pull/4797))
- Open PRs/issues on TruLens, AutoGen, Opik, Langfuse, Giskard, and a community-authored PR on openai-python
- `evalport-sdk` published to both PyPI and npm at v1.0.0
- CI (`.github/workflows/ci.yml`) validates schemas, runs the TS/Python/CLI test suites, and — as of the change below — publishes releases via OIDC, not stored tokens

## The one thing actually outstanding: publish v1.1.0

The repo's SDK sources are at v1.1.0 (Benchmark Hub + `evalport run` CLI) and a GitHub Release for `v1.1.0` already exists, but the registries still only serve v1.0.0 — see [issue #7](https://github.com/adhabnr-ux/evalport/issues/7) for the full history. `ci.yml` no longer needs `PYPI_TOKEN`/`NPM_TOKEN` secrets at all (switched to OIDC Trusted Publishing) — what's left is a one-time registry-side setup, ~5 minutes, that only an account owner logged into pypi.org/npmjs.com can do:

**PyPI** — [pypi.org/manage/project/evalport-sdk/settings/publishing](https://pypi.org/manage/project/evalport-sdk/settings/publishing/) → Add a new GitHub publisher:
| Field | Value |
|---|---|
| Owner | `adhabnr-ux` |
| Repository name | `evalport` |
| Workflow name | `ci.yml` |
| Environment name | `pypi` |

**npm** — [npmjs.com/package/evalport-sdk/access](https://www.npmjs.com/package/evalport-sdk/access) → Trusted Publisher → GitHub Actions:
| Field | Value |
|---|---|
| Organization or user | `adhabnr-ux` |
| Repository | `evalport` |
| Workflow filename | `ci.yml` |
| Environment name | *(leave blank)* |

Once both are set, no new release is needed — just re-run the failed jobs on the existing v1.1.0 release run: https://github.com/adhabnr-ux/evalport/actions/runs/31428654795 (or `gh run rerun 31428654795 --failed`).

## Ongoing, not one-time

- Watch for maintainer replies on open PRs/issues (TruLens #2697, AutoGen #8009, Opik #7798, Langfuse #15930/#16110, Giskard, openai-python #3619) and respond substantively — this is continuous, not a checklist item to "complete."
- Watch Discussions #8–#11 for community input on the open RFC topics.
