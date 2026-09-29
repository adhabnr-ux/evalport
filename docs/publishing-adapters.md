# Publishing adapter packages to PyPI

Every standalone adapter under `adapters/<name>/` can be published to PyPI as
its own package (`pip install <name>`) by
[`.github/workflows/publish-adapter.yml`](../.github/workflows/publish-adapter.yml).
It uses **OIDC Trusted Publishing**, the same mechanism `ci.yml`'s
`publish-pypi` job already uses for `evalport-sdk` (see
[`MANUAL-ACTIONS.md`](../MANUAL-ACTIONS.md)). There is no `PYPI_TOKEN` or any
other API token anywhere: PyPI trusts a short-lived OIDC token that GitHub
issues to this repo's `publish-adapter.yml` workflow, in a named GitHub
Environment.

Check what is published at any time:

```
python scripts/list_adapter_publish_status.py            # pypi.org
python scripts/list_adapter_publish_status.py --testpypi # test.pypi.org
```

## How the workflow works

| | |
|---|---|
| Trigger 1 | push of a tag `adapter/<adapter-dir>/v<version>`, e.g. `adapter/beeai-openeval-adapter/v0.1.0` -> publishes to **PyPI** |
| Trigger 2 | `workflow_dispatch` (Actions tab -> "Publish adapter" -> Run workflow) with inputs `adapter` (directory name) and `repository` (`testpypi` default, or `pypi`), plus an optional `python-version` (default `3.12`) |
| Job `build` | resolves `adapters/<name>` (fails if it doesn't exist or `project.name` differs from the directory name); for a tag, fails unless the tag version equals `pyproject.toml`'s version; installs `-e sdk/python` and `-e adapters/<name>[test]`; runs `pytest tests/` (must pass); `python -m build`; `twine check --strict`; rejects any `name @ git+...` direct reference in the built metadata (PyPI refuses those, and `twine check` does not catch it); installs the built wheel in a clean venv and imports it; uploads `dist/` as an artifact |
| Job `publish` | `needs: build`; `environment: pypi` or `testpypi`; `permissions: id-token: write`; `pypa/gh-action-pypi-publish@release/v1` with `packages-dir: dist/` (and `repository-url: https://test.pypi.org/legacy/` for TestPyPI) |

It never runs on pull requests or branch pushes.

## One-time setup (maintainer, by hand)

These steps need an account owner logged in to the registries; nothing in the
repo can do them.

### 1. GitHub: environments

Repo **Settings -> Environments**:

- **`pypi`** already exists (it is what `ci.yml`'s `publish-pypi` job uses for
  `evalport-sdk`). If it has a *Deployment branches and tags* rule, add a tag
  rule for `adapter/*/v*` (and allow the branch you run `workflow_dispatch`
  from, normally `main`), otherwise the adapter `publish` job will be refused
  by the environment before it ever reaches PyPI.
- **Create `testpypi`** (New environment -> name exactly `testpypi`).
- Optional but recommended: on both, add **Required reviewers** (yourself) so
  every publish waits for a click in the Actions UI. The `build` job (tests,
  build, checks) still runs unattended; only the upload waits.

### 2. PyPI: a pending publisher per adapter project

None of the adapter projects exist on PyPI yet (all 65 names were available
when this was written), so each one needs a **pending publisher**, which turns
into a normal trusted publisher on the first successful upload.

On <https://pypi.org/manage/account/publishing/> -> *Add a new pending
publisher* -> **GitHub** tab, fill in:

| Field | Value |
|---|---|
| PyPI Project Name | the adapter's `project.name`, e.g. `beeai-openeval-adapter` (always identical to its directory name under `adapters/`) |
| Owner | `adhabnr-ux` |
| Repository name | `evalport` |
| Workflow name | `publish-adapter.yml` |
| Environment name | `pypi` |

Notes:

- The existing `evalport-sdk` publisher (workflow `ci.yml`) does **not** cover
  adapters: a trusted publisher is bound to one project *and* one workflow
  file, so every adapter project needs its own entry naming
  `publish-adapter.yml`.
- A pending publisher does not reserve the name. If someone else registers
  the name first, it is gone, so register publishers for the adapters you are
  about to release, then release them soon after.
- After the first upload, the project appears under *Your projects*, and the
  publisher is visible (and editable) at
  `https://pypi.org/manage/project/<name>/settings/publishing/`.

### 3. TestPyPI (optional, for a dry run)

TestPyPI is a separate site with separate accounts. On
<https://test.pypi.org/manage/account/publishing/> add the same pending
publisher, but with **Environment name `testpypi`**. Then:

1. Actions -> *Publish adapter* -> Run workflow -> `adapter` =
   `beeai-openeval-adapter`, `repository` = `testpypi`.
2. Check `https://test.pypi.org/project/beeai-openeval-adapter/`.
3. `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ beeai-openeval-adapter`
   (the extra index is needed because `evalport-sdk` lives on real PyPI).

A version uploaded to TestPyPI cannot be re-uploaded there either, but that
does not affect PyPI.

## Per-release steps

1. Bump `version` in `adapters/<name>/pyproject.toml` (skip for the very first
   release: every adapter currently declares `0.1.0`, which has never been
   published). Merge that to `main`.
2. Tag the merged commit and push the tag:

   ```
   git checkout main && git pull
   git tag adapter/<name>/v<version>          # e.g. adapter/beeai-openeval-adapter/v0.1.0
   git push origin adapter/<name>/v<version>
   ```

3. Watch Actions -> *Publish adapter*. If the `pypi` environment has required
   reviewers, approve the `publish` job.
4. Verify: `python scripts/list_adapter_publish_status.py` (or
   `https://pypi.org/project/<name>/`), then `pip install <name>` in a fresh
   venv.
5. After an adapter's **first** release, update its README's *Install*
   section from the `git+https://...#subdirectory=` command to
   `pip install <name>` (and the corresponding row in the top-level README if
   it shows an install command).

PyPI versions are immutable: a version that was uploaded (even if later
deleted) can never be uploaded again. If a release is broken, bump the
version and tag again.

If a tag's version doesn't match `pyproject.toml`, the `build` job fails
before anything is uploaded; delete the tag (`git push origin
:refs/tags/adapter/<name>/v<version>`), fix, and re-tag.

If an adapter's test extras don't install on Python 3.12 (the workflow's
default, required by `giskard-openeval-adapter`), publish it with
`workflow_dispatch`, `repository` = `pypi`, and a `python-version` override
instead of a tag.

## Recommended first batch

Register pending publishers for and release these first:

| Adapter | Why first |
|---|---|
| `beeai-openeval-adapter` | Blocking adoption: the BeeAI Framework maintainer will only open the example PR in `i-am-bee/beeai-framework` once it is on PyPI ("I'd open the example PR after beeai-openeval-adapter is published to PyPI"). |
| `daimax-openeval-adapter` | Built at the daimax-appbench maintainer's request as a standalone package ([open-daimax/daimax-appbench#9](https://github.com/open-daimax/daimax-appbench/issues/9)). Needed the packaging fix described below before PyPI would accept it at all. |
| `eval-ai-library-openeval-adapter` | Built at the Eval-ai-library owner's request that it live here ([meshkovQA/Eval-ai-library#2](https://github.com/meshkovQA/Eval-ai-library/issues/2)); its users currently have only a git-URL install. |
| `trulens-openeval-adapter` | Upstream connector PR truera/trulens#2757 is pending; a PyPI package gives TruLens users an install path that doesn't depend on it. |
| `langfuse-openeval-adapter` | Langfuse is one of the most widely used LLM observability/eval platforms among the adapters' targets (a judgement call, not a measured download count). |

All five were run through the workflow's `build` job steps locally on Python
3.12 (tests, build, `twine check --strict`, direct-reference check, clean-venv
wheel import) and pass.

`ares-openeval-adapter` is deliberately not in this batch: the same adapter
was merged into IBM/ares as `experimental-plugins/ares-openeval-adapter`
(IBM/ares#583), so check with the ARES maintainers before claiming that
project name on PyPI from this repo.

The `build` job runs each adapter's own test suite and refuses to publish if
it fails, so an adapter whose tests are currently red (for example because a
test hard-codes an older spec version string) must be fixed before it can be
released.

## Packaging notes

- **No direct references in dependencies or extras.** PyPI's upload endpoint
  rejects any `Requires-Dist` that is a PEP 508 direct reference
  (`name @ git+https://...`), including ones behind an extra, with
  *"Can't have direct dependency"* (pypi/warehouse
  `warehouse/forklift/metadata.py`). `twine check` does not detect this, so the
  workflow checks for it explicitly. If an upstream is not on PyPI, document
  installing it from git in the README instead of declaring it as an extra
  (see `daimax-openeval-adapter`).
- **License metadata (follow-up, not yet required).** 66 of 67 adapters use
  the table form `license = {text = "Apache-2.0"}` (65 also carry a
  `License ::` classifier). Current setuptools builds them fine but warns that
  the table form is deprecated (it says builds stop being supported by
  2027-02-18) and that license classifiers are deprecated. Before then,
  switch them to the SPDX form `ares-openeval-adapter` already uses
  (`license = "Apache-2.0"`, no `License ::` classifier,
  `requires = ["setuptools>=77.0.3"]`).