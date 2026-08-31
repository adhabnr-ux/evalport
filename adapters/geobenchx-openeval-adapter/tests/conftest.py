"""Puts a real, cloned GeoBenchX checkout's repo root on sys.path for tests.

This adapter has no runtime dependency on the `geobenchx` package (see the
`geobenchx` extra in pyproject.toml), but its tests are meaningfully stronger
if they exercise the *real* GeoBenchX pydantic classes (Task, Solution, Step,
TaskSet, ScoreValues, TaskLabels) rather than hand-rolled stand-ins -- so the
test-only `geobenchx` extra (numpy/pandas/pydantic) plus a real GeoBenchX
checkout are what's expected to be available when running this test suite.

Set GEOBENCHX_REPO to the checkout's path if it isn't a sibling of this
adapter's own directory (e.g. `/tmp/build_geobenchx/geobenchx_src`, the
default this repo's own CI/dev setup uses).
"""
import os
import sys

_DEFAULT_REPO = "/tmp/build_geobenchx/geobenchx_src"
_REPO = os.environ.get("GEOBENCHX_REPO", _DEFAULT_REPO)

if os.path.isdir(_REPO) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)
