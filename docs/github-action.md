# Validate EvalPort documents in GitHub: Action and pre-commit hook

This repo ships two ways to check EvalPort documents (suites, test cases,
graders, result sets) in another repository, both backed by the same
`evalport-validate` CLI from the Python SDK (`sdk/python/openeval/cli.py`),
which uses the reference validator (`openeval.validate`):

- a **GitHub Action** (`action.yml` at the repo root), which fails the job and
  annotates invalid files on the pull request;
- a **pre-commit hook** (`.pre-commit-hooks.yaml` at the repo root), which runs
  on staged files before each commit.

## GitHub Action

```yaml
# .github/workflows/evalport.yml
name: EvalPort
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: adhabnr-ux/evalport@main
        with:
          files: |
            evals/**/*.evalport.json
            results/*.json
```

`@main` tracks the latest commit. This repo has no release tag that includes
the Action yet. Once one exists, pin to it (`adhabnr-ux/evalport@vX.Y.Z`) or to
a full commit SHA so that upstream changes can't change your CI unexpectedly.
With the default `sdk-version: local`, the validator is installed from the
same ref as the Action, so pinning the Action also pins the validator.

### Inputs

| Input | Default | Description |
|---|---|---|
| `files` | `**/*.evalport.json` | Files, directories or glob patterns, separated by spaces or newlines, relative to the workspace. `**` recurses. Directories are searched recursively for `*.json`. Paths containing spaces are not supported. **If a pattern matches no files, the step fails.** |
| `type` | `auto` | `auto` detects the type of each file: `results` means a result set, `test_cases`/`test_cases_file` a suite, `input` a test case, and `type`+`id` a grader. You can also set `suite`, `testcase`, `grader` or `resultset` to apply one type to every file. |
| `allow-unknown` | `false` | **Proposed ([Discussion #108](https://github.com/adhabnr-ux/evalport/discussions/108)).** Validation is strict by default: a property the schema doesn't define, outside `metadata` (and `provider.extra`, `params`, `summary.by_grader` entries), is an `UNKNOWN_FIELD` error. `true` passes `--allow-unknown`, which accepts such properties. Use it only for documents produced against a newer minor spec version than the validator implements. |
| `python-version` | `3.11` | Passed to `actions/setup-python`. The validator is installed into a private venv under `$RUNNER_TEMP`, so it doesn't affect your job's own Python packages. |
| `sdk-version` | `local` | `local` installs `sdk/python` from the Action's own checkout (`${{ github.action_path }}`). `pypi` installs the latest `evalport-sdk` from PyPI. Any other value installs that exact PyPI version (`evalport-sdk==<value>`). See the caveat below. |

The Action runs `evalport-validate --format github`, so every error becomes a
workflow-command annotation. GitHub shows these in the job log and attaches
them to the file in the pull request's "Files changed" view:

```text
evals/basic.evalport.json: valid (suite)
::error file=evals/broken.evalport.json,line=2,title=EvalPort::$: not valid JSON: Expecting value (line 2, column 9) [INVALID_JSON]
::error file=evals/nightly-run.evalport.json,title=EvalPort::$.results[0].grader_results[0].score: must be in [0,1] or null [OUT_OF_RANGE]
3 files checked: 1 valid, 2 invalid
```

Each annotation has the form `<JSON path>: <message> [<error code>]`, where
the error codes are the ones `openeval.validate` returns. For files that
can't be read or parsed, the codes are `NOT_FOUND`, `READ_ERROR`,
`INVALID_JSON` (with the line number) and `UNKNOWN_TYPE`.

### PyPI caveat (`sdk-version: pypi`)

The `evalport-validate` CLI is new. **`evalport-sdk` 1.3.1, the latest version
on PyPI as of this change, doesn't include it.** With `sdk-version: pypi` or
`sdk-version: 1.3.1`, the Action stops with this error:

```text
::error title=EvalPort::evalport-sdk 1.3.1 (sdk-version: pypi) does not include the evalport-validate CLI. Use sdk-version: local, or a release newer than 1.3.1.
```

Keep the default `local` until an `evalport-sdk` release that includes the CLI
is on PyPI.

### Marketplace

`action.yml` includes `branding` (icon and color) so the Action can be listed
on the GitHub Marketplace later. It isn't listed yet. Until it is, use the
`uses: adhabnr-ux/evalport@<ref>` form shown above.

## pre-commit hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/adhabnr-ux/evalport
    rev: main   # pin a tag or full commit SHA once a release includes the hook
    hooks:
      - id: evalport-validate
        # Optional: the hook only matches *.evalport.json files by default
        # (files: \.evalport\.json$). Override `files` to change that:
        # files: ^evals/.*\.json$
        # To force a document type instead of auto-detecting it:
        # args: [--type, resultset]
        # To accept properties the schema doesn't define (proposed, #108):
        # args: [--allow-unknown]
```

pre-commit warns when `rev` is a branch name such as `main`, because a branch
can change. Run `pre-commit autoupdate` to pin a commit or tag.

The hook is `language: python` with `entry: evalport-validate` and
`pass_filenames: true`. It only runs on files whose names match `files`, and
it fails the commit if any of them is invalid.

**How pre-commit installs it.** For a `language: python` hook, pre-commit
clones this repo at `rev` and runs `pip install .` in the **repo root**. The
Python SDK lives in `sdk/python/`, so this repo also has a small root
`pyproject.toml`. It builds the same `openeval` package from `sdk/python/`
under a separate, unpublished distribution name (`evalport-pre-commit-hook`).
Without that file the hook fails with `Directory '.' is not installable`
(verified with `pre-commit try-repo`). Because of this setup, the hook runs
the validator at exactly the pinned `rev` and doesn't depend on any PyPI
release, so the PyPI caveat above doesn't apply to it.

To try the hook without changing your config:

```bash
pre-commit try-repo https://github.com/adhabnr-ux/evalport evalport-validate --files path/to/suite.evalport.json
```

## Running the same check locally

```bash
pip install ./sdk/python            # from a checkout of this repo
evalport-validate 'evals/**/*.evalport.json'
```

See [`sdk/python/README.md`](../sdk/python/README.md#validate-from-the-command-line-evalport-validate)
for all CLI options and exit codes.

## How this repo tests them

`.github/workflows/action-selftest.yml` runs whenever `action.yml`,
`pyproject.toml`, `.pre-commit-hooks.yaml` or `sdk/python/**` changes. It
checks three things:

- `uses: ./` passes on `examples/*.json`.
- `uses: ./` fails on a known-invalid conformance fixture and on a pattern
  that matches no files. Both steps use `continue-on-error`, and a later step
  checks that their outcome was `failure`. It also fails on an unknown-field
  fixture in strict mode and passes on the same file with
  `allow-unknown: 'true'` (proposed, Discussion #108).
- `pre-commit try-repo` passes on valid files and fails on the invalid
  fixture.
