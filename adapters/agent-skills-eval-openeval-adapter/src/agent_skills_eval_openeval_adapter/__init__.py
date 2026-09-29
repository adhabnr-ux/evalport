"""agent-skills-eval <-> EvalPort adapter.

Standalone converter between the on-disk run artifacts written by
agent-skills-eval (https://github.com/darkrishabh/agent-skills-eval) --
`grading.json`, `meta.json`, `timing.json`, `outputs/response.txt`,
`tool_calls.json` -- and the EvalPort interchange format
(https://github.com/adhabnr-ux/evalport).

Why this exists as a standalone package: agent-skills-eval's maintainer asked
for exactly this shape rather than an in-repo dependency --
https://github.com/darkrishabh/agent-skills-eval/issues/34#issuecomment-5888398803
("A standalone adapter in EvalPort is the direction we'd prefer... No
first-party export or dependency is planned at this stage."). agent-skills-eval
is an npm package with no Python API to import, so this adapter reads its
on-disk JSON artifacts instead of calling into it -- the same "read the public
artifact shape from the outside" approach the autogen-openeval-adapter package
takes toward AutoGen's task/result objects.

Mapping, read directly from agent-skills-eval's own source
(`src/types.ts`, `src/grade.ts`) and from `docs/artifact-contract.md`
(merged in https://github.com/darkrishabh/agent-skills-eval/pull/37,
commit 72c3c65c6472f5d6cbe1f957b33a4d218a3da807):

| agent-skills-eval                                    | EvalPort                                   |
|-------------------------------------------------------|---------------------------------------------|
| one run directory (`grading.json` + friends)           | one `Result`                                |
| `grading.json.assertion_results[].text`                 | `GraderResult.metadata.text` (verbatim)     |
| `grading.json.assertion_results[].passed`               | `GraderResult.passed`, and `score` (1.0/0.0)|
| `grading.json.assertion_results[].evidence`              | `GraderResult.reason`                       |
| `grading.json.summary`                                  | `Result.metadata.agent_skills_eval.summary` |
| `outputs/response.txt`                                  | `Result.actual_output`                      |
| `timing.json.duration_ms`                                | `Result.duration_ms`                        |
| run mode (`with_skill` / `without_skill`)                | `ResultSet.metadata.agent_skills_eval.mode` and each `Result.metadata.agent_skills_eval.mode` (declared once per `ResultSet`, mirroring how `isolation` is declared once per `ResultSet` in the EvalPort spec) |

**The one caveat the upstream maintainer specifically flagged, and how this
adapter honors it:** `grading.json`'s `assertion_results` rows do not carry a
type field, and are a positional concatenation of LLM-judge rubric results
(one per the eval's `assertions: string[]`) followed by deterministic
tool-assertion results (one per `tool_assertions: ToolAssertion[]`) -- see
`gradeOutputs()` in `src/grade.ts`, `[...rubricResults, ...toolResults]`. This
adapter classifies each row as `llm_judge` (EvalPort `GraderResult.type`) only
when the caller also supplies the original eval definition (its `assertions`
and `tool_assertions` arrays) so the split can be computed exactly; without
it, every row is reported as `custom` (never assumed `llm_judge`) and flagged
`metadata.kind_inferred = False`, matching the maintainer's explicit request
not to "label every row llm_judge."
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "load_run_artifacts",
    "iter_eval_dirs",
    "__version__",
]
__version__ = "0.1.0"

PathLike = Union[str, "Path"]


def _classify_assertion_results(
    assertion_results: Sequence[Dict[str, Any]],
    eval_def: Optional[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], str, bool]]:
    """Split `assertion_results` into (row, kind, kind_inferred) triples.

    `kind` is `"llm_judge"` or `"tool_assertion"`. See the module docstring
    for why this can only be done exactly when `eval_def` is supplied, and
    why the fallback is `"tool_assertion"` (mapped to EvalPort's `custom`
    grader type), never `"llm_judge"`.
    """
    n = len(assertion_results)
    if eval_def is not None:
        n_rubric = len(eval_def.get("assertions") or [])
        n_tool = len(eval_def.get("tool_assertions") or [])
        if n_rubric + n_tool == n:
            kinds = ["llm_judge"] * n_rubric + ["tool_assertion"] * n_tool
            return [(row, kind, True) for row, kind in zip(assertion_results, kinds)]
        # eval_def doesn't line up with this grading.json (stale eval_def, a
        # hand-edited grading.json, ...) -- fall through to the conservative
        # default below rather than silently mis-splitting on a row count
        # that doesn't match.
    return [(row, "tool_assertion", False) for row in assertion_results]


def to_openeval(
    runs: Iterable[Dict[str, Any]],
    *,
    suite_id: str,
    run_id: str,
    mode: str,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    runner_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Export a collection of agent-skills-eval run artifacts to an EvalPort ResultSet.

    `runs` is any iterable of dicts, one per completed eval run (typically
    produced by `load_run_artifacts()`), each with:

    - `test_case_id` (str, required): a stable id for the eval case.
    - `grading` (dict, required): the parsed contents of that run's
      `grading.json` (`assertion_results` + `summary`).
    - `eval_def` (dict, optional): the original `AgentSkillsEval` object
      (`assertions`, `tool_assertions`) for exact assertion-row typing --
      see the module docstring.
    - `actual_output` (str, optional): the parsed contents of
      `outputs/response.txt`.
    - `duration_ms` (number, optional): from `timing.json`.
    - `metadata` (dict, optional): extra provenance to preserve verbatim
      (e.g. `{"skill": ..., "eval_slug": ..., "source_path": ...}`).

    `mode` (`"with_skill"` or `"without_skill"`) is declared once for the
    whole `ResultSet`, per the original proposal
    (https://github.com/darkrishabh/agent-skills-eval/issues/34) -- run
    `to_openeval()` once per mode and give each call its own `run_id` if you
    have both. Mixing runs from both modes into one call is not rejected
    (agent-skills-eval's own artifact layout does not prevent it), but the
    single `mode` value would then be misleading for whichever runs don't
    match it, so this is left as a caller responsibility rather than
    silently validated here.

    An `actual_output` beginning with `"ERROR:"` (agent-skills-eval's
    documented convention for a captured provider error, per
    `docs/artifact-contract.md`) additionally sets `Result.error`.

    Returns a plain dict conforming to the EvalPort ResultSet schema. Pass it
    to `openeval.validate.validate_result_set()` to confirm compliance.
    """
    if started_at is None:
        from datetime import datetime, timezone

        started_at = datetime.now(timezone.utc).isoformat()

    results: List[Dict[str, Any]] = []
    for run in runs:
        test_case_id = str(run["test_case_id"])
        grading = run.get("grading") or {}
        assertion_results = grading.get("assertion_results") or []
        eval_def = run.get("eval_def")
        classified = _classify_assertion_results(assertion_results, eval_def)

        grader_results: List[Dict[str, Any]] = []
        for i, (row, kind, inferred) in enumerate(classified):
            passed = bool(row.get("passed"))
            grader_results.append(
                {
                    "grader_id": f"assertion_{i}",
                    "type": "llm_judge" if kind == "llm_judge" else "custom",
                    "score": 1.0 if passed else 0.0,
                    "passed": passed,
                    "reason": row.get("evidence") or None,
                    "metadata": {
                        "text": row.get("text", ""),
                        "kind": kind,
                        "kind_inferred": inferred,
                    },
                }
            )

        summary = grading.get("summary") or {}
        pass_rate = summary.get("pass_rate")
        if pass_rate is not None:
            passed_overall = pass_rate == 1
        elif grader_results:
            passed_overall = all(g["passed"] for g in grader_results)
        else:
            passed_overall = True  # matches grade.ts summarize(): 0 assertions -> pass_rate 1

        result: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "passed": bool(passed_overall),
            "grader_results": grader_results,
        }

        actual_output = run.get("actual_output")
        if actual_output is not None:
            result["actual_output"] = actual_output
            if actual_output.startswith("ERROR:"):
                result["error"] = {
                    "type": "provider_error",
                    "message": actual_output[len("ERROR:") :].strip(),
                }

        if run.get("duration_ms") is not None:
            result["duration_ms"] = int(run["duration_ms"])

        namespaced_meta: Dict[str, Any] = {"mode": mode}
        if summary:
            namespaced_meta["summary"] = dict(summary)
        extra_meta = run.get("metadata")
        if extra_meta:
            namespaced_meta.update(dict(extra_meta))
        result["metadata"] = {"agent_skills_eval": namespaced_meta}

        results.append(result)

    if not results:
        # EvalPort's spec requires a non-empty $.results
        # (openeval.validate.validate_result_set rejects `[]`). Raised here,
        # at the point the caller can act on it, rather than handed back as
        # an unvalidatable ResultSet.
        raise ValueError(
            "to_openeval() received zero runs; EvalPort's ResultSet.results must be "
            "non-empty (spec/SPEC.md)"
        )

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    rs_summary = {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": (passed_count / total) if total else 0,
    }

    runner: Dict[str, str] = {"name": "agent-skills-eval"}
    if runner_version:
        runner["version"] = runner_version

    return {
        "version": OPENEVAL_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": completed_at or started_at,
        "runner": runner,
        "results": results,
        "summary": rs_summary,
        "metadata": {"openeval": {"source": "agent-skills-eval"}, "agent_skills_eval": {"mode": mode}},
    }


def from_openeval(result_set: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Reconstruct agent-skills-eval-shaped grading data from an EvalPort ResultSet.

    Returns `{test_case_id: {"mode": ..., "grading": {"assertion_results": [...], "summary": {...}}}}`.

    This is the inverse of `to_openeval()` for the fields agent-skills-eval's
    own `AssertionResult` type defines (`text`, `passed`, `evidence`).
    EvalPort-side bookkeeping this adapter adds (`score`, `kind`,
    `kind_inferred`) has no corresponding field in `AssertionResult` and is
    not written back -- a documented, one-way loss, not a silent one. If a
    `Result.metadata.agent_skills_eval.summary` recorded by `to_openeval()`
    isn't present (e.g. this `ResultSet` came from another producer), the
    summary is recomputed from the reconstructed rows instead of being left
    absent.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in result_set.get("results", []):
        meta = (r.get("metadata") or {}).get("agent_skills_eval") or {}
        mode = meta.get("mode")

        assertion_results = []
        for gr in r.get("grader_results", []):
            gmeta = gr.get("metadata") or {}
            assertion_results.append(
                {
                    "text": gmeta.get("text", ""),
                    "passed": bool(gr.get("passed")),
                    "evidence": gr.get("reason") or "",
                }
            )

        summary = meta.get("summary")
        if summary is None:
            passed = sum(1 for a in assertion_results if a["passed"])
            total = len(assertion_results)
            summary = {
                "passed": passed,
                "failed": total - passed,
                "total": total,
                "pass_rate": (passed / total) if total else 1,
            }

        out[str(r.get("test_case_id"))] = {
            "mode": mode,
            "grading": {"assertion_results": assertion_results, "summary": summary},
        }
    return out


def load_run_artifacts(
    run_dir: PathLike,
    *,
    test_case_id: Optional[str] = None,
    eval_def: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Read one agent-skills-eval run directory into a `to_openeval()`-ready dict.

    `run_dir` is a `<mode>/` directory as laid out by
    `docs/artifact-contract.md` (e.g.
    `<workspace>/iteration-N/<eval-slug>/with_skill/`), i.e. the directory
    that directly contains `grading.json`. Reads `grading.json` (required),
    and `outputs/response.txt` / `timing.json` when present (both optional --
    a run that errored before producing output may be missing one or both).

    `test_case_id` defaults to `run_dir.parent.name` (the eval-slug
    directory) when not given; pass an explicit id when running this over
    a multi-skill workspace, where eval slugs are not unique on their own.
    """
    run_dir = Path(run_dir)
    grading_path = run_dir / "grading.json"
    if not grading_path.exists():
        raise FileNotFoundError(
            f"no grading.json in {run_dir} -- per docs/artifact-contract.md, every "
            "completed eval run directory writes one"
        )
    grading = json.loads(grading_path.read_text(encoding="utf-8"))

    run: Dict[str, Any] = {
        "test_case_id": test_case_id or run_dir.parent.name,
        "grading": grading,
    }
    if eval_def is not None:
        run["eval_def"] = eval_def

    response_path = run_dir / "outputs" / "response.txt"
    if response_path.exists():
        run["actual_output"] = response_path.read_text(encoding="utf-8")

    timing_path = run_dir / "timing.json"
    extra_meta: Dict[str, Any] = {"source_path": str(run_dir)}
    if timing_path.exists():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        if isinstance(timing.get("duration_ms"), (int, float)):
            run["duration_ms"] = timing["duration_ms"]
        if "total_tokens" in timing:
            extra_meta["total_tokens"] = timing["total_tokens"]
    run["metadata"] = extra_meta
    return run


def iter_eval_dirs(workspace: PathLike) -> Iterator[Path]:
    """Yield every run directory under `workspace` that contains a `grading.json`.

    Walks the tree rather than assuming a fixed nesting depth, per
    `docs/artifact-contract.md`'s "Locate a skill's output" guidance -- this
    covers the CLI's `iteration-N/<eval-slug>/<mode>/` layout, the
    multi-skill `iteration-N/<skill-slug>/<eval-slug>/<mode>/` layout, and
    the SDK's flat `<skill-slug>/<eval-slug>/<mode>/` layout uniformly.
    """
    workspace = Path(workspace)
    for grading_path in sorted(workspace.rglob("grading.json")):
        yield grading_path.parent
