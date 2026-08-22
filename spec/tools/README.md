# EvalPort Spec Tools

Reference tooling that supports the spec but isn't part of either SDK.
Currently one tool: the suite/ResultSet signature verifier.

## `verify_signature.py`

Resolves [Discussion #8](https://github.com/adhabnr-ux/evalport/discussions/8)
("Suite/result signing for integrity verification") — see
`spec/SPEC.md`'s Extension Mechanism → Suite/ResultSet Signing for the
normative convention this implements, and this section for how to actually
use it.

### The problem this solves

`benchmarks/` (and any other publicly-hosted EvalPort suite) can be
downloaded from many places other than a direct `git clone` of this repo —
mirrors, package registries, someone's copy in their own project. Nothing
about the EvalPort document format itself lets a consumer answer "was this
file genuinely published by this project, byte-for-byte, or was it
modified somewhere along the way?" This tool answers that question for any
artifact this project's CI has signed.

### Install

```bash
pip install sigstore
```

That's the only dependency — `verify_signature.py` has no other
requirements beyond the Python standard library and `sigstore` (which
brings in `cryptography` and a few other transitive deps).

### Verify a signed benchmark suite

Every signed artifact ships as two files: the artifact itself (e.g.
`benchmarks/gsm8k/gsm8k.json`) and a detached bundle alongside it, named
`<filename>.sigstore.json` (e.g. `benchmarks/gsm8k/gsm8k.json.sigstore.json`).
Once `.github/workflows/ci.yml`'s `sign-benchmarks` job has produced these
for a real release (see that job's own comments for where the bundles are
published — attached to the GitHub Release, not committed to `main`),
verify a suite you've downloaded like this:

```bash
python3 spec/tools/verify_signature.py verify benchmarks/gsm8k/gsm8k.json \
  --cert-identity-regex '^https://github\.com/adhabnr-ux/evalport/\.github/workflows/ci\.yml@refs/tags/.*$' \
  --cert-oidc-issuer https://token.actions.githubusercontent.com
```

The regex form (`--cert-identity-regex`) accepts a signature from a
release build cut from *any* tag of this repo's `ci.yml` workflow. Use
`--cert-identity` with an exact string instead if you want to pin to one
specific release:

```bash
python3 spec/tools/verify_signature.py verify benchmarks/gsm8k/gsm8k.json \
  --cert-identity 'https://github.com/adhabnr-ux/evalport/.github/workflows/ci.yml@refs/tags/v1.2.0' \
  --cert-oidc-issuer https://token.actions.githubusercontent.com
```

Exit code `0` means the file is byte-for-byte what this repo's release CI
published and signed. Exit code `1` means it isn't (wrong content, wrong
signer, or no valid signature at all) — treat that suite as untrusted.
Exit code `2` is a usage error (bad arguments); exit code `3` means the
`sigstore` package isn't installed.

**Always pass an identity check.** `--unsafe-skip-identity-check` proves a
file has *some* valid Sigstore signature, not that *this project* signed
it — anyone can get a valid Fulcio certificate for their own GitHub
identity and sign anything. It exists for debugging a bundle in isolation,
not as a real integrity check; the script requires you to pass it
explicitly (or an identity check) rather than defaulting to it, precisely
so this mistake has to be made on purpose.

### What's actually checked

1. The bundle's signature is valid over the artifact's exact bytes (no
   canonicalization — see the convention note below).
2. The signing certificate chains to Sigstore's public root of trust
   (fetched live via [TUF](https://theupdateframework.io/); needs network
   access unless you pass `--offline` with a pre-fetched local trust
   root).
3. The certificate was issued for the identity you specified — this is
   what makes "signed" mean "signed by *this project's CI*" rather than
   "signed by someone."
4. The bundle's [Rekor](https://docs.sigstore.dev/logging/overview/)
   transparency-log entry is present and internally consistent.

Full detail, including why raw bytes are signed instead of a canonical
JSON form, and why Sigstore instead of PGP or a project-managed key, is in
`spec/SPEC.md`'s Extension Mechanism → Suite/ResultSet Signing section and
in the reasoning laid out in
[Discussion #8](https://github.com/adhabnr-ux/evalport/discussions/8).

### Testing this tool itself

`tests/test_verify_signature.py` is a full pytest suite verified against
real, independently-published Sigstore bundles (not synthetic data) — see
that file's module docstring and `tests/fixtures/NOTICE.md` for exactly
what's vendored and why. Run it the same way any other test suite in this
repo runs:

```bash
pip install sigstore pytest
python3 -m pytest spec/tools/tests/ -v
```

This is wired into CI as the `verify-signature-tool` job (see
`.github/workflows/ci.yml`), running on every push and PR — independent of
the `sign-benchmarks` job, which only runs on a real release and needs
this script to already be correct.

### Signing is optional, always

Nothing about validating an EvalPort document — `validate_suite()`,
`validate_result_set()`, the JSON Schemas, the conformance suite — depends
on or checks for a signature. An unsigned suite is exactly as spec-valid
as a signed one; signing exists specifically for the "did this get
tampered with in transit" threat model, which mostly matters for publicly
redistributed suites like `benchmarks/`, not for a ResultSet you generated
in your own CI and never published anywhere else.
