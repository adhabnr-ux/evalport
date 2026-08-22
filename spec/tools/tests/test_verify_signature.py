"""Tests for spec/tools/verify_signature.py.

These are NOT tests against synthetic or self-signed data. Every "valid
bundle" fixture here is a real Sigstore bundle, independently produced and
published by a different project, vendored verbatim (see
tests/fixtures/NOTICE.md for exact provenance and license). Two distinct
real scenarios are covered:

  - tests/fixtures/github-actions-signed.whl(.sigstore.json): a genuine
    artifact signed by trailofbits/rfc8785.py's real GitHub Actions release
    workflow, verified against Sigstore's PRODUCTION instance. This is the
    exact scenario EvalPort's own sign-benchmarks CI job produces: a
    GitHub Actions OIDC identity, Fulcio-issued cert, Rekor-logged
    signature.
  - tests/fixtures/plaintext-personal-identity.txt(.sigstore.json): a
    small text file signed by a human's personal GitHub OAuth identity,
    verified against Sigstore's STAGING instance (this specific fixture
    predates a production-only convention in the upstream project and is
    a staging-instance bundle; --staging is what makes verifying it
    correct, not a relaxation of what's being checked).

Why not test against a bundle this repo's own CI produced? Because as of
this writing, .github/workflows/ci.yml's sign-benchmarks job has not yet
run against a real release -- and minting a NEW Sigstore signature requires
an interactive OIDC login (a browser-based identity provider flow) that a
sandboxed, non-interactive test environment cannot perform. Verifying
against real, independently-issued, already-published bundles is what
proves this script's verification logic is correct; it is honest about not
being a test of this specific repo's CI configuration, which can only be
confirmed once that job runs for real (see the module docstring in
verify_signature.py for the same caveat, and benchmarks/README.md's
"Verifying suite integrity" section for what a real consumer does once it
has).

Every test in this file that expects success requires real network access
(sigstore-python fetches the current Sigstore trust root live via TUF).
Tests are not skipped when offline -- a network failure should be visible
as a real failure, not silently masked, since "can this verify a real
signature" is exactly what's under test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verify_signature import (  # noqa: E402
    DEFAULT_BUNDLE_SUFFIX,
    build_identity_policy,
    default_bundle_path,
    verify_artifact,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"

GITHUB_ARTIFACT = FIXTURES / "github-actions-signed.whl"
GITHUB_BUNDLE = FIXTURES / "github-actions-signed.whl.sigstore.json"
GITHUB_IDENTITY = (
    "https://github.com/trailofbits/rfc8785.py/.github/workflows/"
    "release.yml@refs/tags/v0.1.2"
)
GITHUB_ISSUER = "https://token.actions.githubusercontent.com"

PERSONAL_ARTIFACT = FIXTURES / "plaintext-personal-identity.txt"
PERSONAL_BUNDLE = FIXTURES / "plaintext-personal-identity.txt.sigstore.json"
PERSONAL_IDENTITY = "william@yossarian.net"
PERSONAL_ISSUER = "https://github.com/login/oauth"

pytestmark = pytest.mark.online  # every test here needs real network access


# --------------------------------------------------------------------------
# default_bundle_path / naming convention
# --------------------------------------------------------------------------


def test_default_bundle_path_appends_suffix():
    assert default_bundle_path(Path("gsm8k.json")) == Path(
        f"gsm8k.json{DEFAULT_BUNDLE_SUFFIX}"
    )
    assert DEFAULT_BUNDLE_SUFFIX == ".sigstore.json"


def test_default_bundle_path_preserves_directory():
    assert default_bundle_path(Path("benchmarks/gsm8k/gsm8k.json")) == Path(
        f"benchmarks/gsm8k/gsm8k.json{DEFAULT_BUNDLE_SUFFIX}"
    )


def test_verify_artifact_finds_bundle_by_convention(tmp_path):
    # Copy the real artifact + bundle into a temp dir under the
    # conventional naming, and verify with no --bundle passed at all.
    artifact = tmp_path / "suite.json"
    bundle = tmp_path / f"suite.json{DEFAULT_BUNDLE_SUFFIX}"
    shutil.copy(GITHUB_ARTIFACT, artifact)
    shutil.copy(GITHUB_BUNDLE, bundle)

    outcome = verify_artifact(
        artifact,
        cert_identity=GITHUB_IDENTITY,
        cert_oidc_issuer=GITHUB_ISSUER,
    )
    assert outcome.ok, outcome.message
    assert outcome.bundle == bundle


# --------------------------------------------------------------------------
# Real GitHub Actions OIDC-signed artifact, production Sigstore instance
# --------------------------------------------------------------------------


def test_github_actions_bundle_verifies_with_exact_identity():
    outcome = verify_artifact(
        GITHUB_ARTIFACT,
        GITHUB_BUNDLE,
        cert_identity=GITHUB_IDENTITY,
        cert_oidc_issuer=GITHUB_ISSUER,
    )
    assert outcome.ok, outcome.message


def test_github_actions_bundle_verifies_with_regex_identity_any_tag():
    outcome = verify_artifact(
        GITHUB_ARTIFACT,
        GITHUB_BUNDLE,
        cert_identity_regex=(
            r"^https://github\.com/trailofbits/rfc8785\.py/"
            r"\.github/workflows/release\.yml@refs/tags/.*$"
        ),
        cert_oidc_issuer=GITHUB_ISSUER,
    )
    assert outcome.ok, outcome.message


def test_github_actions_bundle_rejects_wrong_exact_identity():
    outcome = verify_artifact(
        GITHUB_ARTIFACT,
        GITHUB_BUNDLE,
        cert_identity=(
            "https://github.com/some/other-repo/.github/workflows/"
            "release.yml@refs/tags/v9.9.9"
        ),
        cert_oidc_issuer=GITHUB_ISSUER,
    )
    assert not outcome.ok
    assert "do not match" in outcome.message


def test_github_actions_bundle_rejects_wrong_regex_identity():
    outcome = verify_artifact(
        GITHUB_ARTIFACT,
        GITHUB_BUNDLE,
        # The regex EvalPort's own CI would actually use -- correctly
        # rejected here, since this fixture was never signed by evalport.
        cert_identity_regex=(
            r"^https://github\.com/adhabnr-ux/evalport/"
            r"\.github/workflows/ci\.yml@refs/tags/.*$"
        ),
        cert_oidc_issuer=GITHUB_ISSUER,
    )
    assert not outcome.ok
    assert "do not match regex" in outcome.message


def test_github_actions_bundle_rejects_wrong_oidc_issuer():
    outcome = verify_artifact(
        GITHUB_ARTIFACT,
        GITHUB_BUNDLE,
        cert_identity=GITHUB_IDENTITY,
        cert_oidc_issuer="https://accounts.google.com",
    )
    assert not outcome.ok


def test_tampered_artifact_is_rejected(tmp_path):
    tampered = tmp_path / "github-actions-signed.whl"
    original_bytes = bytearray(GITHUB_ARTIFACT.read_bytes())
    # Flip one byte in the middle of the file -- enough to change the
    # digest without corrupting the zip container structure itself.
    flip_at = len(original_bytes) // 2
    original_bytes[flip_at] ^= 0xFF
    tampered.write_bytes(bytes(original_bytes))

    outcome = verify_artifact(
        tampered,
        GITHUB_BUNDLE,
        cert_identity=GITHUB_IDENTITY,
        cert_oidc_issuer=GITHUB_ISSUER,
    )
    assert not outcome.ok
    assert "digest mismatch" in outcome.message or "failed" in outcome.message


def test_truncated_artifact_is_rejected(tmp_path):
    truncated = tmp_path / "github-actions-signed.whl"
    original = GITHUB_ARTIFACT.read_bytes()
    truncated.write_bytes(original[: len(original) // 2])

    outcome = verify_artifact(
        truncated,
        GITHUB_BUNDLE,
        cert_identity=GITHUB_IDENTITY,
        cert_oidc_issuer=GITHUB_ISSUER,
    )
    assert not outcome.ok


# --------------------------------------------------------------------------
# Real personal-identity artifact, staging Sigstore instance
# --------------------------------------------------------------------------


def test_personal_identity_bundle_verifies_on_staging():
    outcome = verify_artifact(
        PERSONAL_ARTIFACT,
        PERSONAL_BUNDLE,
        cert_identity=PERSONAL_IDENTITY,
        cert_oidc_issuer=PERSONAL_ISSUER,
        staging=True,
    )
    assert outcome.ok, outcome.message


def test_personal_identity_bundle_rejects_wrong_email():
    outcome = verify_artifact(
        PERSONAL_ARTIFACT,
        PERSONAL_BUNDLE,
        cert_identity="not-the-real-signer@example.com",
        cert_oidc_issuer=PERSONAL_ISSUER,
        staging=True,
    )
    assert not outcome.ok


def test_personal_identity_bundle_fails_against_production_instance():
    # This bundle was issued against Sigstore's staging root of trust, not
    # production. Verifying it against production must fail on the
    # certificate chain, not silently succeed -- staging and production
    # are different, non-interchangeable trust roots, and a verifier that
    # accepted a staging-issued cert as production-valid would be a real
    # security hole (staging certs are issued with no real identity
    # assurance, precisely so they're safe to use for testing).
    outcome = verify_artifact(
        PERSONAL_ARTIFACT,
        PERSONAL_BUNDLE,
        cert_identity=PERSONAL_IDENTITY,
        cert_oidc_issuer=PERSONAL_ISSUER,
        staging=False,
    )
    assert not outcome.ok


# --------------------------------------------------------------------------
# Missing files / malformed input -- must fail cleanly, not crash
# --------------------------------------------------------------------------


def test_missing_artifact_fails_cleanly():
    outcome = verify_artifact(
        FIXTURES / "does-not-exist.json",
        GITHUB_BUNDLE,
        cert_identity=GITHUB_IDENTITY,
    )
    assert not outcome.ok
    assert "not found" in outcome.message


def test_missing_bundle_fails_cleanly():
    outcome = verify_artifact(
        GITHUB_ARTIFACT,
        FIXTURES / "does-not-exist.sigstore.json",
        cert_identity=GITHUB_IDENTITY,
    )
    assert not outcome.ok
    assert "not found" in outcome.message


def test_malformed_bundle_fails_cleanly(tmp_path):
    bad_bundle = tmp_path / "bad.sigstore.json"
    bad_bundle.write_text("{ this is not valid sigstore bundle json")

    outcome = verify_artifact(
        GITHUB_ARTIFACT,
        bad_bundle,
        cert_identity=GITHUB_IDENTITY,
    )
    assert not outcome.ok
    assert "could not parse" in outcome.message


def test_no_identity_policy_specified_is_a_clear_error():
    with pytest.raises(ValueError, match="no identity policy specified"):
        build_identity_policy(
            cert_identity=None,
            cert_identity_regex=None,
            cert_oidc_issuer=None,
            unsafe_skip_identity_check=False,
        )


def test_verify_artifact_surfaces_missing_identity_policy_as_outcome():
    # Via the public verify_artifact() entry point (not the lower-level
    # build_identity_policy() helper), the same misconfiguration should
    # come back as a normal failed VerificationOutcome, not an unhandled
    # exception -- verify_artifact() is meant to be safe to call in a loop
    # over many files without a caller having to wrap every call in
    # try/except.
    outcome = verify_artifact(GITHUB_ARTIFACT, GITHUB_BUNDLE)
    assert not outcome.ok
    assert "no identity policy specified" in outcome.message


def test_unsafe_skip_identity_check_verifies_signature_only():
    # Signature/chain/Rekor validity still genuinely checked -- only the
    # "who signed it" check is skipped. Confirms this still rejects a
    # tampered artifact even with identity checking off.
    outcome = verify_artifact(
        GITHUB_ARTIFACT, GITHUB_BUNDLE, unsafe_skip_identity_check=True
    )
    assert outcome.ok, outcome.message


def test_unsafe_skip_identity_check_still_rejects_tampering(tmp_path):
    tampered = tmp_path / "github-actions-signed.whl"
    original_bytes = bytearray(GITHUB_ARTIFACT.read_bytes())
    original_bytes[0] ^= 0xFF
    tampered.write_bytes(bytes(original_bytes))

    outcome = verify_artifact(
        tampered, GITHUB_BUNDLE, unsafe_skip_identity_check=True
    )
    assert not outcome.ok


# --------------------------------------------------------------------------
# CLI entry point (subprocess, exercising real argv parsing + exit codes)
# --------------------------------------------------------------------------


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    script = Path(__file__).resolve().parent.parent / "verify_signature.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_cli_verify_success_exit_code_zero():
    result = _run_cli(
        "verify",
        str(GITHUB_ARTIFACT),
        "--bundle",
        str(GITHUB_BUNDLE),
        "--cert-identity",
        GITHUB_IDENTITY,
        "--cert-oidc-issuer",
        GITHUB_ISSUER,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_cli_verify_failure_exit_code_one():
    result = _run_cli(
        "verify",
        str(GITHUB_ARTIFACT),
        "--bundle",
        str(GITHUB_BUNDLE),
        "--cert-identity",
        "wrong@example.com",
        "--cert-oidc-issuer",
        GITHUB_ISSUER,
    )
    assert result.returncode == 1


def test_cli_no_identity_flag_exit_code_two():
    result = _run_cli(
        "verify",
        str(GITHUB_ARTIFACT),
        "--bundle",
        str(GITHUB_BUNDLE),
    )
    assert result.returncode == 2
    assert "required" in result.stderr


def test_cli_regex_identity_any_tag():
    result = _run_cli(
        "verify",
        str(GITHUB_ARTIFACT),
        "--bundle",
        str(GITHUB_BUNDLE),
        "--cert-identity-regex",
        (
            r"^https://github\.com/trailofbits/rfc8785\.py/"
            r"\.github/workflows/release\.yml@refs/tags/.*$"
        ),
        "--cert-oidc-issuer",
        GITHUB_ISSUER,
    )
    assert result.returncode == 0, result.stdout + result.stderr
