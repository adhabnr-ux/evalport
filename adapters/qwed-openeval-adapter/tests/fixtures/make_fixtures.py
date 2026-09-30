"""Regenerate the Verification Context fixtures in this directory.

Needs `pip install "qwed>=7.1.0,<8"` (the fixtures were generated with
qwed 7.2.1). Uses only QWED's supported public surface:

* `qwed_sdk` exports (`VerificationContextDocument`, `VerificationContext`,
  `Interpretation`, `Proof`, `Evidence`, `Decision`, `Admission`,
  `Formalization`) for the two VERIFIED documents. `.verified()` computes
  the real `proof_ref` commitment. These are constructed documents, not the
  output of running a QWED engine.
* The `qwed context from-diagnostic` CLI for the UNVERIFIABLE and BLOCKED
  documents, fed DiagnosticResult JSON. The BLOCKED one is a VERIFIED
  diagnostic with no attestation token, which qwed 7.2.1 demotes to BLOCKED
  (fail-closed).

Run from this directory: `python make_fixtures.py`.
"""
import json
import os
import subprocess
import sys
import tempfile

from qwed_sdk import (
    Admission,
    Decision,
    Evidence,
    Formalization,
    Interpretation,
    Proof,
    VerificationContext,
    VerificationContextDocument,
    validate_document,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def _write(name, doc):
    validate_document(doc)
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _cli(diagnostic, query, verifier):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(diagnostic, f)
        path = f.name
    try:
        out = subprocess.run(
            [os.path.join(os.path.dirname(sys.executable), "qwed"), "context", "from-diagnostic",
             "--diagnostic-file", path, "--query", query, "--verifier", verifier,
             "--verifier-version", "7.2.1"],
            check=True, capture_output=True, text=True,
        )
    finally:
        os.unlink(path)
    return json.loads(out.stdout)


def main():
    sql = VerificationContextDocument.verified(
        formal_statement="SELECT id, total FROM orders WHERE customer_id = 42;",
        formalization=Formalization(
            source_query="Total of every order placed by customer 42",
            translator="example-llm",
        ),
        context=VerificationContext(
            interpretation=Interpretation(dialect="sqlite", parser_version="sqlglot"),
            proof=Proof(verifier="sql", verifier_version="7.2.1"),
            evidence=Evidence(payload={"agent_message": "Read-only SELECT; no injection pattern found.",
                                       "tables": ["orders"], "statement_type": "SELECT"}),
            decision=Decision(admission=Admission.ADMIT),
        ),
    )
    _write("vc_sql_verified_admit.json", sql.to_dict())

    code = VerificationContextDocument.verified(
        formal_statement="import os\nos.system(user_input)",
        context=VerificationContext(
            interpretation=Interpretation(language="python", policy_version="code-security-v1"),
            proof=Proof(verifier="code", verifier_version="7.2.1"),
            evidence=Evidence(payload={"agent_message": "Proven unsafe: unsanitized shell exec via os.system.",
                                       "finding": "os.system", "line": 2}),
            decision=Decision(admission=Admission.DENY),
        ),
    )
    _write("vc_code_verified_deny.json", code.to_dict())

    fact = _cli(
        {"status": "UNVERIFIABLE", "agent_message": "No retrievable evidence supports or refutes the claim.",
         "developer_fields": {"constraint_id": "fact.insufficient_evidence"}, "proof_ref": None,
         "is_authoritative": False},
        query="The Eiffel Tower was completed in 1889.",
        verifier="fact",
    )
    _write("vc_fact_unverifiable.json", fact)

    blocked = _cli(
        {"status": "VERIFIED", "agent_message": "Equality holds.", "developer_fields": {},
         "proof_ref": "sha256:" + "0" * 64},
        query="2 + 2 = 4",
        verifier="math",
    )
    assert blocked["verdict"] == "BLOCKED", blocked["verdict"]
    _write("vc_math_blocked_no_attestation.json", blocked)


if __name__ == "__main__":
    main()
