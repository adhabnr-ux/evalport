#!/usr/bin/env python3
"""verify_signature.py -- reference verifier for the EvalPort suite/ResultSet
signing convention.

Resolves https://github.com/adhabnr-ux/evalport/discussions/8. See
spec/SPEC.md, Extension Mechanism -> Suite/ResultSet Signing, for the
normative description of the convention this script implements; this
docstring is a practical summary, not the source of truth.

THE CONVENTION, IN BRIEF
-------------------------
  - Signing is OPTIONAL and entirely out-of-band: it changes nothing about
    the suite or ResultSet document itself (no schema change, no new
    `metadata` key), and trust in an EvalPort document never depends on a
    signature being present. It exists for one specific threat model: "I
    downloaded this suite from somewhere other than a direct git clone of
    the publisher's repo -- did anyone tamper with it in transit or at
    rest?" -- which matters most for publicly-hosted benchmark suites (see
    benchmarks/README.md's "Verifying suite integrity" section) and matters
    little for a ResultSet you generated yourself in your own CI.
  - A signed artifact is any published file (today: every JSON file under
    benchmarks/) plus a DETACHED Sigstore bundle sitting alongside it,
    named "<filename>.sigstore.json" -- e.g. "gsm8k.json.sigstore.json"
    next to "gsm8k.json". The core document is never modified to carry a
    signature.
  - The bundle signs the RAW PUBLISHED BYTES of the artifact -- no JSON
    canonicalization step (no JCS/RFC 8785). This means a byte-identical
    re-serialization with different whitespace needs a fresh signature,
    but it also means no canonicalizer implementation is required in any
    consuming language, and there is no room for two implementations'
    canonicalizers to quietly disagree about what was "really" signed.
  - Signing uses Sigstore's keyless flow: this repo's GitHub Actions
    release workflow exchanges its OIDC token for a short-lived Fulcio
    certificate -- no long-lived private key exists anywhere for anyone to
    leak or rotate. The signature, the certificate, and a Rekor
    transparency-log inclusion proof are bundled together in the
    .sigstore.json file. This is the same OIDC trust root already used for
    this repo's PyPI/npm Trusted Publishing (see .github/workflows/ci.yml's
    publish-pypi / publish-npm jobs) -- extending that same trust to
    benchmark-suite signing is a small conceptual step, not new
    infrastructure, and needs no key-management or revocation story.

WHAT THIS SCRIPT VERIFIES
--------------------------
  1. The bundle's signature is cryptographically valid over the artifact's
     exact bytes (not a canonicalized or re-serialized form).
  2. The signing certificate chains to the public Sigstore root of trust
     (fetched live via TUF -- requires network access, unless --offline is
     passed with a local trust root).
  3. The certificate was issued for the IDENTITY policy the caller
     specifies. Signature validity alone only proves "someone with some
     Sigstore-recognized OIDC identity signed this" -- without an identity
     check, that is not a useful integrity guarantee, since anyone can get
     a Fulcio cert for their own identity and sign anything. This script
     requires the caller to state an expected identity; it refuses to
     verify signature-validity-only ("--cert-identity" or
     "--cert-identity-regex", plus "--cert-oidc-issuer", are required
     unless "--unsafe-skip-identity-check" is explicitly passed).
  4. The bundle's Rekor transparency-log entry is present and consistent
     (sigstore-python's Verifier does this internally as part of bundle
     verification; a bundle with a missing or inconsistent log entry fails
     verification here too).

HONEST SCOPE NOTE ON TESTING
------------------------------
This script's own test suite (spec/tools/tests/test_verify_signature.py)
verifies it against REAL, independently-published Sigstore bundles --
including a genuine GitHub Actions OIDC-signed artifact -- vendored from
the sigstore-python project's own test fixtures (Apache-2.0; see
spec/tools/tests/fixtures/NOTICE.md for exact provenance). It is not
tested against a bundle this project's own CI produced, because as of this
writing .github/workflows/ci.yml's sign-benchmarks job has not yet run on
a real release, and minting a *new* Sigstore signature requires an
interactive OIDC login (a browser-based identity-provider flow) that this
development environment cannot perform non-interactively or headlessly.
Verifying against real, independently-issued bundles from a different
GitHub Actions repository is a meaningful test of this script's own
correctness (it proves the verification logic -- signature check, Fulcio
chain validation, Rekor inclusion, identity-policy matching, and rejection
of tampered input or a wrong identity policy -- genuinely works against
production Sigstore infrastructure); it does not by itself prove that this
repo's CI job is correctly configured, which can only be confirmed once
that job has actually run.

USAGE
-----
  # Verify a suite signed by this repo's own release CI (once it exists):
  python3 verify_signature.py verify benchmarks/gsm8k/gsm8k.json \\
      --cert-identity-regex '^https://github\\.com/adhabnr-ux/evalport/\\.github/workflows/ci\\.yml@refs/tags/.*$' \\
      --cert-oidc-issuer https://token.actions.githubusercontent.com

  # Verify an explicit bundle path against an exact identity:
  python3 verify_signature.py verify some/file.json --bundle some/file.json.sig.json \\
      --cert-identity 'https://github.com/OWNER/REPO/.github/workflows/release.yml@refs/tags/v1.2.3' \\
      --cert-oidc-issuer https://token.actions.githubusercontent.com

Exit codes: 0 = verified and identity policy matched. 1 = verification
failed (tampered artifact, invalid bundle, or identity mismatch) -- this is
the normal "no" answer, not a bug. 2 = usage error (bad arguments, missing
files). 3 = the `sigstore` package isn't installed.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from sigstore.errors import Error as SigstoreError
    from sigstore.models import Bundle
    from sigstore.verify import Verifier
    from sigstore.verify import policy as sigstore_policy
except ImportError:  # pragma: no cover - exercised via a subprocess test
    sys.stderr.write(
        "error: the 'sigstore' package is required (pip install sigstore) "
        "but is not installed in this Python environment.\n"
    )
    sys.exit(3)


DEFAULT_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
DEFAULT_BUNDLE_SUFFIX = ".sigstore.json"


class RegexIdentity:
    """A sigstore VerificationPolicy that matches a certificate's Subject
    Alternative Name (SAN) against a regular expression instead of an exact
    string.

    `sigstore.verify.policy.Identity` (the library's built-in policy) only
    supports exact-string SAN matching, which is too rigid for EvalPort's
    real use case: verifying "this suite was signed by a release build of
    adhabnr-ux/evalport, for *any* tag", not one single hardcoded tag. This
    class implements the same `VerificationPolicy` protocol
    (`verify(cert) -> None`, raising on mismatch) that every built-in
    sigstore-python policy implements, so it composes with
    `sigstore.verify.policy.AllOf` / `AnyOf` exactly like a built-in policy
    would.
    """

    def __init__(self, pattern: str):
        self._pattern = re.compile(pattern)

    def verify(self, cert) -> None:  # noqa: ANN001 - cert is sigstore's Certificate
        from cryptography import x509

        san_ext = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value
        all_sans = set(san_ext.get_values_for_type(x509.RFC822Name))
        all_sans.update(san_ext.get_values_for_type(x509.UniformResourceIdentifier))

        if not any(self._pattern.match(san) for san in all_sans):
            raise SigstoreError(
                f"Certificate's SANs do not match regex {self._pattern.pattern!r}; "
                f"actual SANs: {sorted(all_sans)}"
            )


@dataclass
class VerificationOutcome:
    ok: bool
    artifact: Path
    bundle: Path
    message: str

    def __bool__(self) -> bool:
        return self.ok


def default_bundle_path(artifact: Path) -> Path:
    """The conventional detached-bundle path for a given artifact:
    "<filename><DEFAULT_BUNDLE_SUFFIX>" alongside it, e.g.
    "gsm8k.json" -> "gsm8k.json.sigstore.json"."""
    return artifact.with_name(artifact.name + DEFAULT_BUNDLE_SUFFIX)


def build_identity_policy(
    *,
    cert_identity: Optional[str],
    cert_identity_regex: Optional[str],
    cert_oidc_issuer: Optional[str],
    unsafe_skip_identity_check: bool,
):
    """Builds the sigstore VerificationPolicy the caller asked for.

    Exactly one of (cert_identity, cert_identity_regex,
    unsafe_skip_identity_check) must be meaningfully set; this is enforced
    by the CLI's argparse mutually-exclusive group, but re-checked here so
    this function is also safe to call directly from other Python code
    (e.g. a future `evalport run` integration) without going through the
    CLI parser.
    """
    if unsafe_skip_identity_check:
        return sigstore_policy.UnsafeNoOp()

    if cert_identity_regex:
        parts = [RegexIdentity(cert_identity_regex)]
        if cert_oidc_issuer:
            parts.append(sigstore_policy.OIDCIssuer(cert_oidc_issuer))
        return (
            sigstore_policy.AllOf(parts) if len(parts) > 1 else parts[0]
        )

    if cert_identity:
        return sigstore_policy.Identity(
            identity=cert_identity, issuer=cert_oidc_issuer
        )

    raise ValueError(
        "no identity policy specified: pass --cert-identity or "
        "--cert-identity-regex, or explicitly pass "
        "unsafe_skip_identity_check=True to verify signature validity only "
        "(NOT RECOMMENDED -- proves *someone* signed it, not *who*)"
    )


def verify_artifact(
    artifact_path: Path,
    bundle_path: Optional[Path] = None,
    *,
    cert_identity: Optional[str] = None,
    cert_identity_regex: Optional[str] = None,
    cert_oidc_issuer: Optional[str] = DEFAULT_OIDC_ISSUER,
    unsafe_skip_identity_check: bool = False,
    offline: bool = False,
    staging: bool = False,
) -> VerificationOutcome:
    """Verify `artifact_path` against its detached Sigstore bundle.

    Returns a VerificationOutcome rather than raising, so callers (e.g. a
    batch-verification loop over every file in benchmarks/) can collect
    results for every artifact instead of stopping at the first failure.
    """
    artifact_path = Path(artifact_path)
    bundle_path = Path(bundle_path) if bundle_path else default_bundle_path(
        artifact_path
    )

    if not artifact_path.is_file():
        return VerificationOutcome(
            False, artifact_path, bundle_path, f"artifact not found: {artifact_path}"
        )
    if not bundle_path.is_file():
        return VerificationOutcome(
            False,
            artifact_path,
            bundle_path,
            f"signature bundle not found: {bundle_path} "
            f"(expected alongside the artifact, named "
            f"'<artifact-filename>{DEFAULT_BUNDLE_SUFFIX}')",
        )

    try:
        policy = build_identity_policy(
            cert_identity=cert_identity,
            cert_identity_regex=cert_identity_regex,
            cert_oidc_issuer=cert_oidc_issuer,
            unsafe_skip_identity_check=unsafe_skip_identity_check,
        )
    except ValueError as e:
        return VerificationOutcome(False, artifact_path, bundle_path, str(e))

    try:
        bundle_bytes = bundle_path.read_bytes()
        bundle = Bundle.from_json(bundle_bytes)
    except Exception as e:  # sigstore/json raise a variety of error types here
        return VerificationOutcome(
            False,
            artifact_path,
            bundle_path,
            f"could not parse signature bundle {bundle_path}: {e}",
        )

    verifier = Verifier.staging(offline=offline) if staging else Verifier.production(
        offline=offline
    )

    try:
        artifact_bytes = artifact_path.read_bytes()
        verifier.verify_artifact(artifact_bytes, bundle, policy)
    except Exception as e:
        return VerificationOutcome(
            False,
            artifact_path,
            bundle_path,
            f"verification failed for {artifact_path}: {e}",
        )

    return VerificationOutcome(
        True,
        artifact_path,
        bundle_path,
        f"OK: {artifact_path} is genuinely signed and matches the given identity policy",
    )


def _add_common_verify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("artifact", type=Path, help="path to the file to verify")
    p.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help=(
            "path to the detached .sigstore.json bundle. Defaults to "
            "'<artifact><suffix>' alongside the artifact "
            f"(suffix: '{DEFAULT_BUNDLE_SUFFIX}')."
        ),
    )
    identity_group = p.add_mutually_exclusive_group(required=False)
    identity_group.add_argument(
        "--cert-identity",
        default=None,
        help=(
            "exact expected certificate SAN (Subject Alternative Name), e.g. "
            "'https://github.com/OWNER/REPO/.github/workflows/ci.yml@refs/tags/v1.0.0' "
            "for a GitHub Actions release build, or an email address for a "
            "human-signed artifact."
        ),
    )
    identity_group.add_argument(
        "--cert-identity-regex",
        default=None,
        help=(
            "regular expression the certificate SAN must fully match "
            "(via re.match), e.g. "
            r"'^https://github\.com/OWNER/REPO/\.github/workflows/ci\.yml@refs/tags/.*$' "
            "to accept a release build from any tag."
        ),
    )
    identity_group.add_argument(
        "--unsafe-skip-identity-check",
        action="store_true",
        help=(
            "verify signature validity only, without checking WHO signed it. "
            "NOT RECOMMENDED for any real integrity check -- anyone can obtain "
            "a valid Sigstore signature for their own identity. Provided for "
            "debugging a bundle in isolation, not for verifying trust."
        ),
    )
    p.add_argument(
        "--cert-oidc-issuer",
        default=DEFAULT_OIDC_ISSUER,
        help=(
            f"expected OIDC issuer that vouched for the signer's identity "
            f"(default: {DEFAULT_OIDC_ISSUER}, GitHub Actions' issuer). "
            "Pass '' to skip the issuer check when using --cert-identity "
            "(not recommended)."
        ),
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help=(
            "verify using a locally cached/bundled trust root instead of "
            "fetching the current one live via TUF. Requires sigstore-python "
            "to already have a usable local trust root; mainly useful for "
            "air-gapped verification, not the default CI/CLI path."
        ),
    )
    p.add_argument(
        "--staging",
        action="store_true",
        help=(
            "verify against Sigstore's staging instance instead of the "
            "public-good production instance. EvalPort's real releases are "
            "always signed against production; this exists for dry-running "
            "the signing/verification pipeline end-to-end against Sigstore's "
            "staging infrastructure before it touches a real release, and "
            "for this tool's own test suite (see spec/tools/tests/)."
        ),
    )


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_signature.py",
        description=(
            "Verify an EvalPort suite/ResultSet artifact against its "
            "detached Sigstore signature bundle. See this file's module "
            "docstring for the full convention this implements."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify_p = sub.add_parser(
        "verify", help="verify one artifact against its signature bundle"
    )
    _add_common_verify_args(verify_p)

    args = parser.parse_args(argv)

    if args.command == "verify":
        if not (
            args.cert_identity
            or args.cert_identity_regex
            or args.unsafe_skip_identity_check
        ):
            parser.error(
                "one of --cert-identity, --cert-identity-regex, or "
                "--unsafe-skip-identity-check is required"
            )

        cert_oidc_issuer = args.cert_oidc_issuer or None

        outcome = verify_artifact(
            args.artifact,
            args.bundle,
            cert_identity=args.cert_identity,
            cert_identity_regex=args.cert_identity_regex,
            cert_oidc_issuer=cert_oidc_issuer,
            unsafe_skip_identity_check=args.unsafe_skip_identity_check,
            offline=args.offline,
            staging=args.staging,
        )

        print(outcome.message)
        return 0 if outcome.ok else 1

    parser.error(f"unknown command: {args.command}")  # pragma: no cover
    return 2


if __name__ == "__main__":
    sys.exit(main())
