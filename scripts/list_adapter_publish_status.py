#!/usr/bin/env python3
"""List every adapter under adapters/ with its local version and PyPI status.

For each adapters/<name>/pyproject.toml this prints the project name, the
version declared locally, and what PyPI's JSON API reports for that project:

  - "not on PyPI"            -> https://pypi.org/pypi/<name>/json returned 404
  - "<latest> (up to date)"  -> the local version is already published
  - "<latest> (local newer)" -> local pyproject version not yet published
  - "<latest> (MISMATCH)"    -> PyPI's latest is newer than local, or PyPI has
                                releases but not this version and it isn't newer

Usage (from the repo root):
    python scripts/list_adapter_publish_status.py             # plain table
    python scripts/list_adapter_publish_status.py --markdown  # markdown table
    python scripts/list_adapter_publish_status.py --testpypi  # query test.pypi.org

Stdlib only (Python 3.11+ for tomllib; falls back to `tomli` if installed).
See docs/publishing-adapters.md for how to publish an adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parent.parent
ADAPTERS_DIR = REPO_ROOT / "adapters"


def _version_key(v: str):
    """Best-effort PEP 440 ordering; uses `packaging` when available."""
    try:
        from packaging.version import Version

        return Version(v)
    except Exception:
        return tuple(int(p) if p.isdigit() else 0 for p in v.split("."))


def pypi_info(project: str, index: str) -> tuple[str, dict | None]:
    """Return (status, json) where status is 'found', 'missing' or an error."""
    url = f"https://{index}/pypi/{project}/json"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return "found", json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing", None
        return f"HTTP {e.code}", None
    except (urllib.error.URLError, TimeoutError) as e:
        return f"error: {getattr(e, 'reason', e)}", None


def describe(local_version: str, status: str, data: dict | None) -> str:
    if status == "missing":
        return "not on PyPI"
    if status != "found" or data is None:
        return status
    latest = data["info"]["version"]
    released = set(data.get("releases", {}))
    if local_version in released:
        return f"{latest} (up to date)" if latest == local_version else f"{latest} (local {local_version} already published)"
    if _version_key(local_version) > _version_key(latest):
        return f"{latest} (local newer)"
    return f"{latest} (MISMATCH: local {local_version})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--markdown", action="store_true", help="print a markdown table")
    ap.add_argument("--testpypi", action="store_true", help="query test.pypi.org instead of pypi.org")
    args = ap.parse_args()
    index = "test.pypi.org" if args.testpypi else "pypi.org"

    rows = []
    for pyproject in sorted(ADAPTERS_DIR.glob("*/pyproject.toml")):
        with pyproject.open("rb") as f:
            project = tomllib.load(f)["project"]
        rows.append((pyproject.parent.name, project["name"], str(project["version"])))

    with ThreadPoolExecutor(max_workers=8) as pool:
        infos = list(pool.map(lambda r: pypi_info(r[1], index), rows))

    out = [(d, v, describe(v, s, j)) for (d, _n, v), (s, j) in zip(rows, infos)]
    published = sum(1 for _, _, st in out if st != "not on PyPI" and not st.startswith(("HTTP", "error")))
    errors = sum(1 for _, _, st in out if st.startswith(("HTTP", "error")))

    header = ("adapter", "version", index)
    if args.markdown:
        print(f"| {header[0]} | {header[1]} | {header[2]} |")
        print("|---|---|---|")
        for row in out:
            print(f"| {row[0]} | {row[1]} | {row[2]} |")
    else:
        w0 = max(len(header[0]), *(len(r[0]) for r in out))
        w1 = max(len(header[1]), *(len(r[1]) for r in out))
        print(f"{header[0]:<{w0}}  {header[1]:<{w1}}  {header[2]}")
        for row in out:
            print(f"{row[0]:<{w0}}  {row[1]:<{w1}}  {row[2]}")
    print(f"\n{len(out)} adapters: {published} on {index}, {len(out) - published - errors} not on {index}"
          + (f", {errors} lookup errors" if errors else ""))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
