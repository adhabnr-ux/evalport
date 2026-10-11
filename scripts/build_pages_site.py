#!/usr/bin/env python3
"""Build the static GitHub Pages site that serves EvalPort's JSON Schemas.

The site layout is derived from the schemas' own ``$id`` values, so every URL
the spec already uses resolves unchanged once the ``evalport.org`` domain is
pointed at GitHub Pages (see docs/schema-hosting.md):

    https://evalport.org/schema/suite.json          -> <out>/schema/suite.json
    https://evalport.org/schema/v1.0.0/suite.json   -> <out>/schema/v1.0.0/suite.json
    https://evalport.org/                           -> <out>/index.html

Inputs (read-only; nothing in the repo is modified):
    spec/schemas/{suite,testcase,grader,resultset}.json   (source of truth)
    docs/landing-page.html                                (becomes index.html)

After building, the script scans the repo for every
``https://evalport.org/schema/<file>.json`` URL (in .json/.md/.ts/.py files
other than this script) and
fails if any of them does not map to a file in the built tree. That check is
what keeps "every existing URL resolves unchanged" true as the spec evolves.

Usage (from the repo root):
    python scripts/build_pages_site.py            # builds into ./_site
    python scripts/build_pages_site.py --out DIR  # builds into DIR

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

SELF = Path(__file__).resolve()
REPO = SELF.parent.parent
SCHEMA_DIR = REPO / "spec" / "schemas"
LANDING_PAGE = REPO / "docs" / "landing-page.html"
SCHEMA_NAMES = ("suite", "testcase", "grader", "resultset")
SCHEMA_HOST = "evalport.org"

# Pinned copies published alongside the "latest" files. SPEC.md "Schema
# Evolution" documents https://evalport.org/schema/v1.0.0/testcase.json as the
# pinned URL form. Until 1.0.0 final is released, v1.0.0/ carries the same
# bytes as the latest files (i.e. it is NOT yet frozen); see
# docs/schema-hosting.md "Pinned paths" for how to freeze it at release time.
PINNED_VERSIONS = ("v1.0.0",)

URL_RE = re.compile(r"https://evalport\.org(/schema/[A-Za-z0-9._/-]+\.json)")
SCAN_SUFFIXES = {".json", ".md", ".ts", ".py"}
SKIP_DIRS = {".git", "node_modules", "_site", "dist", "build", "__pycache__"}


def schema_path_from_id(name: str) -> str:
    """Return the URL path of spec/schemas/<name>.json's $id (e.g. /schema/suite.json)."""
    with open(SCHEMA_DIR / f"{name}.json", encoding="utf-8") as f:
        schema_id = json.load(f)["$id"]
    parsed = urlparse(schema_id)
    if parsed.scheme != "https" or parsed.netloc != SCHEMA_HOST:
        sys.exit(f"error: {name}.json $id {schema_id!r} is not an https://{SCHEMA_HOST}/ URL")
    if not parsed.path.endswith(f"/{name}.json"):
        sys.exit(f"error: {name}.json $id {schema_id!r} does not end in /{name}.json")
    return parsed.path


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for name in SCHEMA_NAMES:
        src = SCHEMA_DIR / f"{name}.json"
        url_path = schema_path_from_id(name)  # e.g. /schema/suite.json
        latest = out / url_path.lstrip("/")
        latest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, latest)  # byte-for-byte copy
        for version in PINNED_VERSIONS:
            pinned = latest.parent / version / latest.name
            pinned.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, pinned)

    shutil.copyfile(LANDING_PAGE, out / "index.html")


def iter_repo_files():
    for path in REPO.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(REPO).parts):
            continue
        if path == SELF:
            continue  # this script's own docstring/examples are not references
        if path.is_file() and path.suffix in SCAN_SUFFIXES:
            yield path


def check_urls(out: Path) -> int:
    """Every https://evalport.org/schema/*.json URL in the repo must exist in the site."""
    found: dict[str, list[str]] = {}
    for path in iter_repo_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for match in URL_RE.finditer(text):
            url_path = match.group(1)
            if "*" in url_path:
                continue
            found.setdefault(url_path, []).append(str(path.relative_to(REPO)))

    missing = 0
    for url_path in sorted(found):
        exists = (out / url_path.lstrip("/")).is_file()
        status = "ok     " if exists else "MISSING"
        print(f"{status} https://{SCHEMA_HOST}{url_path}  ({len(found[url_path])} reference(s))")
        if not exists:
            missing += 1
            for ref in found[url_path]:
                print(f"          referenced in {ref}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", default="_site", help="output directory (default: _site)")
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out

    build(out)
    print(f"Built site in {out}:")
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        print(f"  {path.relative_to(out)}")
    print()
    missing = check_urls(out)
    if missing:
        print(f"\nerror: {missing} evalport.org schema URL(s) referenced in the repo are not in the built site")
        return 1
    print("\nAll evalport.org schema URLs referenced in the repo map to a file in the built site.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
