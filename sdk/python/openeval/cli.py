"""``evalport-validate``: validate EvalPort documents from the command line.

Thin command-line wrapper over :mod:`openeval.validate` so EvalPort documents
can be checked in CI (the repo-root ``action.yml`` GitHub Action) and in
pre-commit (the repo-root ``.pre-commit-hooks.yaml``) without writing any
Python. Zero runtime dependencies, like the rest of the package.

Usage::

    evalport-validate [--type auto|suite|testcase|grader|resultset]
                      [--format text|github|json] [--include GLOB]
                      [--allow-unknown]
                      PATH_OR_GLOB [PATH_OR_GLOB ...]

Each argument is a file, a directory (searched recursively for files whose
name matches ``--include``, default ``*.json``) or a glob pattern (``**``
recurses). Exit status: 0 when every document is valid, 1 when any document
is invalid or could not be read/parsed, 2 on a usage error (including an
argument that matches no files).

Validation is strict by default: a property the schema does not define,
outside a ``metadata`` object, is an ``UNKNOWN_FIELD`` error (issue #107 /
Discussion #108, PROPOSED). ``--allow-unknown`` turns only that check off, for
consumers reading documents from a newer minor spec version.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from .validate import validate_document

DOC_TYPES = ("suite", "testcase", "grader", "resultset")
TYPE_CHOICES = ("auto",) + DOC_TYPES
FORMATS = ("text", "github", "json")
_GLOB_CHARS = set("*?[")


def detect_type(doc: Any) -> Optional[str]:
    """Best-effort guess of an EvalPort document's type from its top-level keys.

    The discriminating field of each document type, per spec/schemas/:
    a ResultSet has ``results``; a Suite has ``test_cases`` (or the
    ``test_cases_file`` alternative the validator also accepts); a TestCase
    has ``input``; a Grader has ``type`` + ``id`` and no ``input``. Checked
    in that order, so a Suite that happens to carry an ``input``-like key is
    still a Suite. Returns None when nothing matches (not an EvalPort
    document, or too malformed to tell) -- callers must pass ``--type``.
    """
    if not isinstance(doc, dict):
        return None
    if "results" in doc:
        return "resultset"
    if "test_cases" in doc or "test_cases_file" in doc:
        return "suite"
    if "input" in doc:
        return "testcase"
    if "type" in doc and "id" in doc:
        return "grader"
    return None


def _has_glob(s: str) -> bool:
    return any(c in _GLOB_CHARS for c in s)


def expand_paths(args: Sequence[str], include: str) -> "tuple[List[str], List[str]]":
    """Expand files/directories/globs into a de-duplicated, ordered file list.

    Returns ``(files, unmatched)`` where ``unmatched`` lists every argument
    that resolved to no files at all (a missing path or an empty glob).
    """
    files: List[str] = []
    seen = set()
    unmatched: List[str] = []

    def add(p: str) -> None:
        key = os.path.normpath(p)
        if key not in seen:
            seen.add(key)
            files.append(p)

    def add_dir(d: str) -> int:
        n = 0
        for root, dirs, names in os.walk(d):
            dirs[:] = sorted(x for x in dirs if not x.startswith("."))
            for name in sorted(names):
                if fnmatch.fnmatch(name, include):
                    add(os.path.join(root, name))
                    n += 1
        return n

    for arg in args:
        if os.path.isdir(arg):
            if add_dir(arg) == 0:
                unmatched.append(arg)
        elif os.path.exists(arg) or not _has_glob(arg):
            # A missing non-glob path is reported per-file (NOT_FOUND) so the
            # user sees exactly which name was wrong, rather than a usage error.
            add(arg)
        else:
            matches = sorted(glob.glob(arg, recursive=True))
            n = 0
            for m in matches:
                if os.path.isdir(m):
                    n += add_dir(m)
                else:
                    add(m)
                    n += 1
            if n == 0:
                unmatched.append(arg)
    return files, unmatched


def _file_error(message: str, code: str, line: Optional[int] = None) -> Dict[str, Any]:
    e: Dict[str, Any] = {"path": "$", "message": message, "code": code}
    if line is not None:
        e["line"] = line
    return e


def check_file(path: str, doc_type: str = "auto", allow_unknown: bool = False) -> Dict[str, Any]:
    """Validate one file. Never raises for bad input; problems become errors."""
    report: Dict[str, Any] = {"file": path, "type": None, "valid": False, "errors": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        report["errors"].append(_file_error("file not found", "NOT_FOUND"))
        return report
    except (OSError, UnicodeDecodeError) as exc:
        report["errors"].append(_file_error(f"could not read file: {exc}", "READ_ERROR"))
        return report
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        report["errors"].append(
            _file_error(f"not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
                        "INVALID_JSON", exc.lineno)
        )
        return report

    t = detect_type(doc) if doc_type == "auto" else doc_type
    if t is None:
        report["errors"].append(_file_error(
            "could not detect the EvalPort document type (expected a suite with "
            "`test_cases`, a result set with `results`, a test case with `input`, "
            "or a grader with `type` and `id`); pass --type to force one",
            "UNKNOWN_TYPE"))
        return report
    report["type"] = t
    result = validate_document(doc, t, allow_unknown=allow_unknown)
    report["valid"] = bool(result.valid)
    report["errors"] = [dict(e) for e in result.errors]
    return report


def _escape_data(s: str) -> str:
    # https://docs.github.com/actions/reference/workflow-commands-for-github-actions
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(s: str) -> str:
    return _escape_data(s).replace(":", "%3A").replace(",", "%2C")


def _format_error(e: Dict[str, Any]) -> str:
    return f"{e['path']}: {e['message']} [{e['code']}]"


def render(reports: List[Dict[str, Any]], fmt: str) -> str:
    n_invalid = sum(1 for r in reports if not r["valid"])
    n_valid = len(reports) - n_invalid
    summary = (f"{len(reports)} file{'s' if len(reports) != 1 else ''} checked: "
               f"{n_valid} valid, {n_invalid} invalid")
    if fmt == "json":
        return json.dumps({"valid": n_invalid == 0, "files": reports}, indent=2)
    lines: List[str] = []
    for r in reports:
        kind = r["type"] or "unknown type"
        if r["valid"]:
            lines.append(f"{r['file']}: valid ({kind})")
            continue
        n = len(r["errors"])
        if fmt == "github":
            for e in r["errors"]:
                props = f"file={_escape_property(r['file'])}"
                if e.get("line") is not None:
                    props += f",line={e['line']}"
                props += ",title=EvalPort"
                lines.append(f"::error {props}::{_escape_data(_format_error(e))}")
        else:
            lines.append(f"{r['file']}: invalid ({kind}), {n} error{'s' if n != 1 else ''}")
            for e in r["errors"]:
                lines.append(f"  {_format_error(e)}")
    lines.append(summary)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalport-validate",
        description="Validate EvalPort documents (suites, test cases, graders, result sets).",
        epilog="Exit status: 0 all valid, 1 any invalid or unreadable, 2 usage error.",
    )
    p.add_argument("paths", nargs="+", metavar="PATH",
                   help="files, directories (searched recursively) or glob patterns (** recurses)")
    p.add_argument("--type", dest="doc_type", choices=TYPE_CHOICES, default="auto",
                   help="document type; 'auto' (default) detects it per file from its top-level keys")
    p.add_argument("--format", dest="fmt", choices=FORMATS, default="text",
                   help="'text' (default), 'github' (workflow-command annotations), or 'json'")
    p.add_argument("--include", default="*.json", metavar="GLOB",
                   help="filename pattern used when searching directories (default: *.json)")
    p.add_argument("--allow-unknown", action="store_true",
                   help="accept properties the schema does not define (no UNKNOWN_FIELD errors); "
                        "for reading documents from a newer minor spec version. Validation is "
                        "strict by default, and strict mode is the one conformance claims use")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    files, unmatched = expand_paths(args.paths, args.include)
    if unmatched:
        for u in unmatched:
            msg = f"no files matched: {u}"
            if args.fmt == "github":
                print(f"::error title=EvalPort::{_escape_data(msg)}", file=sys.stderr)
            else:
                print(f"evalport-validate: {msg}", file=sys.stderr)
        return 2
    reports = [check_file(f, args.doc_type, args.allow_unknown) for f in files]
    print(render(reports, args.fmt))
    return 0 if all(r["valid"] for r in reports) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
