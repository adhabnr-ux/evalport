# EvalPort — Manual Actions Checklist (Updated September 2, 2026)

Nothing is currently outstanding that requires a human with registry or account credentials. This file is kept as a log of what was manual and how it got resolved, in case a similar situation recurs (e.g. a registry credential needs re-linking after an account change).

## Resolved: PyPI/npm publishing (issue #7)

`evalport-sdk` now publishes automatically to both registries via GitHub Actions OIDC Trusted Publishing whenever a GitHub Release is cut — no stored `PYPI_TOKEN`/`NPM_TOKEN` secrets, no manual `twine upload`/`npm publish`.

What made this possible, in order:
1. `ci.yml`'s `publish-pypi`/`publish-npm` jobs were switched from token-based auth to OIDC Trusted Publishing (no credentials in the workflow at all).
2. The registry side of that trust relationship — linking `pypi.org`/`npmjs.com` to this exact repo + workflow file — required an account owner to log in and register it by hand:
   - PyPI: [pypi.org/manage/project/evalport-sdk/settings/publishing](https://pypi.org/manage/project/evalport-sdk/settings/publishing/) → GitHub publisher: owner `adhabnr-ux`, repo `evalport`, workflow `ci.yml`, environment `pypi`.
   - npm: [npmjs.com/package/evalport-sdk/access](https://www.npmjs.com/package/evalport-sdk/access) → Trusted Publisher → GitHub Actions: `adhabnr-ux/evalport`, workflow `ci.yml`, no environment.
   - Both done September 2, 2026.
3. First real release under the new setup (`v1.3.0`) confirmed `publish-pypi` works, but caught a second, unrelated real bug: `publish-npm` failed npm's OIDC provenance check because `sdk/typescript/package.json` had no `repository` field. Fixed and re-released as `v1.3.1` — all CI jobs green, both registries verified live at 1.3.1.

Verify current published versions any time with `curl -s https://registry.npmjs.org/evalport-sdk | jq .'dist-tags.latest'` and https://pypi.org/project/evalport-sdk/.

## Ongoing, not one-time

- Watch for maintainer replies on open outreach threads and respond substantively — this is continuous, not a checklist item to "complete."
- Watch open RFC Discussions for community input.
- If a registry ever again serves a stale version after a release, check the Actions run for that release tag first (`publish-pypi`/`publish-npm` job logs) before assuming a credentials problem — the last two failures here were a missing registry-side Trusted Publisher link and a missing `package.json` field, not expired secrets.
