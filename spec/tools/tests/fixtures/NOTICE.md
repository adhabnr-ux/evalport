# Fixture provenance

The four files in this directory are real, unmodified Sigstore artifacts
vendored from the [`sigstore/sigstore-python`](https://github.com/sigstore/sigstore-python)
project's own test suite (`test/assets/`), used here under that project's
Apache License 2.0 (see `LICENSE` in this directory for the full license
text, copied from the same source repository).

They are used as-is, unmodified, as real-world test fixtures for
`spec/tools/verify_signature.py` — see that script's module docstring and
`spec/tools/tests/test_verify_signature.py`'s module docstring for why
real, independently-published bundles are used here instead of bundles
signed by this project.

## `github-actions-signed.whl` + `.sigstore.json`

- Source path in upstream repo: `test/assets/bundle_v3_github.whl` +
  `test/assets/bundle_v3_github.whl.sigstore`
- What it is: a real wheel artifact built and signed by
  [`trailofbits/rfc8785.py`](https://github.com/trailofbits/rfc8785.py)'s
  own GitHub Actions release workflow, tag `v0.1.2`.
- Certificate identity (Subject Alternative Name):
  `https://github.com/trailofbits/rfc8785.py/.github/workflows/release.yml@refs/tags/v0.1.2`
- OIDC issuer: `https://token.actions.githubusercontent.com` (GitHub
  Actions' own OIDC issuer)
- Sigstore instance: **production** (the public-good instance)
- This is the fixture that exercises exactly the scenario EvalPort's own
  `sign-benchmarks` CI job produces: a GitHub Actions OIDC identity,
  Fulcio-issued short-lived certificate, Rekor-logged signature.

## `plaintext-personal-identity.txt` + `.sigstore.json`

- Source path in upstream repo: `test/assets/bundle_v3.txt` +
  `test/assets/bundle_v3.txt.sigstore`
- What it is: a small text file signed by a sigstore-python maintainer's
  personal GitHub OAuth identity, used upstream as a general-purpose "v3
  bundle format" test fixture.
- Certificate identity (Subject Alternative Name): `william@yossarian.net`
- OIDC issuer: `https://github.com/login/oauth`
- Sigstore instance: **staging** — this specific upstream fixture is a
  staging-instance bundle (confirmed by reading `sigstore-python`'s own
  test suite, `test/unit/verify/test_verifier.py`, which verifies it with
  `Verifier.staging()`), which is why
  `test_verify_signature.py::test_personal_identity_bundle_verifies_on_staging`
  passes `staging=True`. Verifying it against the production instance is
  expected to fail (see
  `test_personal_identity_bundle_fails_against_production_instance`) —
  staging and production are different, non-interchangeable trust roots by
  design, not a bug in either the fixture or the verifier.

## Why vendor these instead of generating fresh fixtures

Minting a *new* Sigstore signature requires an interactive OIDC login (a
browser-based identity-provider flow) that a headless, non-interactive
development environment cannot perform. Verifying these real,
already-published, independently-issued bundles is what proves
`verify_signature.py`'s verification logic — signature validity, Fulcio
certificate chain validation, Rekor transparency-log inclusion, and
identity-policy matching/rejection — is genuinely correct against
production Sigstore infrastructure, not a synthetic stand-in for it.
