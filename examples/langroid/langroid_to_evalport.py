#!/usr/bin/env python3
"""Run Langroid Tasks, record how each one ended, and write EvalPort documents.

Langroid's ``Task.run()`` returns ``None`` when a task ends as ``STALLED`` or
``MAX_TURNS`` (``Task.result`` in langroid/agent/task.py does this on purpose: its
source comment says the result is not known and Langroid does not want to guess it).
The ``StatusCode`` that says *why* the task ended is then dropped. A harness that grades
``run()``'s return value therefore turns "the task never produced an answer" into an
empty string that fails an ``exact_match`` grader: an infrastructure outcome is recorded
as a wrong answer.

This script runs eight small scenarios through Langroid's real ``Task`` machinery
(only the LLM is mocked, with Langroid's own ``MockLM``; no network, no API key),
records the real terminal status of each, and writes a Suite and a ResultSet that use
Validation Rule 6 (``score: null`` = "not verified") for every run whose outcome was
not established. ``--policy naive`` writes the document a harness that grades whatever
``run()`` returned would write, for comparison.

It can also add the ``Result.verdict`` field proposed in EvalPort Discussion #49
(``--proposed-verdict``). That field is NOT part of the spec; see README.md.

Not an official Langroid integration: nothing here has been proposed to, or reviewed
by, the Langroid maintainers.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import langroid as lr
from langroid.agent.chat_document import StatusCode
from langroid.language_models.mock_lm import MockLMConfig
from langroid.utils.configuration import settings

SPEC_VERSION = "1.0.0"
SUITE_ID = "suite_langroid_task_states"
RUN_ID = "run_langroid_001"
GRADER_ID = "gr_exact"
QUESTION = "What is 2+2?"
EXPECTED = "4"
FIXED_STARTED_AT = "2026-10-03T00:00:00Z"

# Statuses whose returned content is, by Langroid's own design, the task's answer.
# DONE: an agent or tool said DONE. FIXED_TURNS: the caller asked for exactly N turns
# (``run(turns=N)``), so the last message is the result *by construction*.
# Everything else (STALLED, MAX_TURNS, TIMEOUT, KILL, ...) ended for a reason other
# than "the task finished", so what was produced (if anything) is not an answer.
JUDGED_STATUSES = frozenset({StatusCode.DONE, StatusCode.FIXED_TURNS})

POLICIES = ("status-aware", "naive")


# --------------------------------------------------------------------------------------
# Running Langroid
# --------------------------------------------------------------------------------------
class RecordingTask(lr.Task):
    """A Task that remembers the status ``result()`` was called with.

    ``Task.run()`` calls ``self.result(status)``. For STALLED / MAX_TURNS / INF_LOOP the
    base implementation returns None, so the status is otherwise lost. Overriding
    ``result()`` is a documented extension point and changes no behaviour.
    """

    last_status: Optional[StatusCode] = None

    def result(self, status: Optional[StatusCode] = None):  # type: ignore[override]
        self.last_status = status
        return super().result(status)


@contextlib.contextmanager
def langroid_settings(**overrides: Any) -> Iterator[None]:
    """Set Langroid's global settings for one run, then restore them."""
    saved = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        setattr(settings, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(settings, k, v)


ResponseFn = Callable[[str], Optional[str]]


@dataclass(frozen=True)
class Scenario:
    """One Task configuration plus the status Langroid is expected to end it with.

    ``expected_status`` is the *name* of a ``StatusCode``, or ``"EXCEPTION"`` when
    ``Task.run`` is expected to raise. The tests assert it against the real run, so a
    Langroid upgrade that changes behaviour fails loudly instead of silently
    producing different documents.
    """

    id: str
    description: str
    expected_status: str
    default_response: str = "Mock response"
    # Builds the mock LLM's response function; gets a dict holding the live task under
    # "task" (needed by the scenario that kills its own task).
    make_response_fn: Optional[Callable[[Dict[str, Any]], ResponseFn]] = None
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    run_kwargs: Dict[str, Any] = field(default_factory=dict)
    global_settings: Dict[str, Any] = field(default_factory=dict)


def _slow(seconds: float) -> Callable[[Dict[str, Any]], ResponseFn]:
    def make(_: Dict[str, Any]) -> ResponseFn:
        def fn(_msg: str) -> str:
            time.sleep(seconds)
            return "still thinking"

        return fn

    return make


def _kill_self(holder: Dict[str, Any]) -> ResponseFn:
    def fn(_msg: str) -> str:
        holder["task"].kill()
        return "partial work"

    return fn


def _raise(_: Dict[str, Any]) -> ResponseFn:
    def fn(_msg: str) -> str:
        raise RuntimeError("provider down")

    return fn


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        "done_correct",
        "The agent answers 'DONE 4'.",
        "DONE",
        default_response="DONE 4",
    ),
    Scenario(
        "done_wrong",
        "The agent answers 'DONE 5': it finished, and it is wrong.",
        "DONE",
        default_response="DONE 5",
    ),
    Scenario(
        "fixed_turns",
        "run(turns=2): the caller asks for exactly two turns; the last message is '4'.",
        "FIXED_TURNS",
        default_response="4",
        run_kwargs={"turns": 2},
    ),
    Scenario(
        "stalled",
        "The agent keeps saying 'thinking...' and never says DONE; Langroid gives up "
        "after 5 stalled steps.",
        "STALLED",
        default_response="thinking...",
    ),
    Scenario(
        "max_turns",
        "As 'stalled', but the global max_turns limit (3) is reached first.",
        "MAX_TURNS",
        default_response="thinking...",
        task_kwargs={"max_stalled_steps": 100},
        global_settings={"max_turns": 3},
    ),
    Scenario(
        "timeout",
        "Each LLM call takes 0.4 s and run(max_time=0.3) is set.",
        "TIMEOUT",
        make_response_fn=_slow(0.4),
        run_kwargs={"max_time": 0.3},
    ),
    Scenario(
        "killed",
        "The agent calls task.kill() on its own task after producing 'partial work'.",
        "KILL",
        make_response_fn=_kill_self,
    ),
    Scenario(
        "agent_raises",
        "The LLM call raises RuntimeError('provider down'); Task.run propagates it.",
        "EXCEPTION",
        make_response_fn=_raise,
    ),
)

SCENARIOS_BY_ID = {s.id: s for s in SCENARIOS}


@dataclass
class Observation:
    """What actually happened when a scenario ran."""

    scenario: Scenario
    status: Optional[str]  # StatusCode name, "EXCEPTION", or None if never recorded
    content: Optional[str]  # run()'s return value's content; None if run() returned None
    run_returned_none: bool
    exception: Optional[str]  # "<type>: <message>" when Task.run raised
    duration_ms: int


def run_scenario(sc: Scenario) -> Observation:
    """Run one scenario through a real ``lr.Task`` with Langroid's MockLM."""
    holder: Dict[str, Any] = {}
    response_fn = sc.make_response_fn(holder) if sc.make_response_fn else (lambda s: None)
    llm = MockLMConfig(
        response_fn=response_fn,
        default_response=sc.default_response,
        cache_config=None,
    )
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="A",
            llm=llm,
            system_message="You answer.",
            use_functions_api=False,
            use_tools=False,
        )
    )
    task = RecordingTask(
        agent,
        interactive=False,
        config=lr.TaskConfig(enable_loggers=False, enable_html_logging=False),
        **sc.task_kwargs,
    )
    holder["task"] = task
    exception: Optional[str] = None
    doc = None
    t0 = time.perf_counter()
    with langroid_settings(quiet=True, cache=False, **sc.global_settings):
        try:
            doc = task.run(QUESTION, **sc.run_kwargs)
        except Exception as e:  # noqa: BLE001 - the point is to record any failure
            exception = f"{type(e).__name__}: {e}"
    duration_ms = int((time.perf_counter() - t0) * 1000)
    if exception is not None:
        status: Optional[str] = "EXCEPTION"
    else:
        status = task.last_status.name if task.last_status is not None else None
    return Observation(
        scenario=sc,
        status=status,
        content=None if doc is None else doc.content,
        run_returned_none=doc is None and exception is None,
        exception=exception,
        duration_ms=duration_ms,
    )


def run_all(scenarios: Tuple[Scenario, ...] = SCENARIOS) -> List[Observation]:
    return [run_scenario(s) for s in scenarios]


# --------------------------------------------------------------------------------------
# Mapping to EvalPort
# --------------------------------------------------------------------------------------
def exact_match(actual: str, expected: str, *, ignore_case: bool = True) -> bool:
    """SPEC ``exact_match``: ``trim_whitespace`` defaults to true; ``ignore_case`` set here."""
    a, e = actual.strip(), expected.strip()
    return a.lower() == e.lower() if ignore_case else a == e


def is_judged(obs: Observation) -> bool:
    """True when the run ended in a state whose output is the task's answer.

    Fail-closed: a status this example has not seen (MAX_COST, MAX_TOKENS, INF_LOOP,
    USER_QUIT, ...), an exception, and a judged status with no content are all treated as
    "not verified".
    """
    if obs.exception is not None or obs.content is None or obs.status is None:
        return False
    return obs.status in {s.name for s in JUDGED_STATUSES}


def unverified_reason(obs: Observation) -> str:
    if obs.exception is not None:
        return f"not verified: Task.run raised {obs.exception}"
    if obs.status is None:
        return "not verified: Langroid recorded no terminal status"
    if obs.run_returned_none:
        return (
            f"not verified: Langroid ended the task as StatusCode.{obs.status} and "
            "Task.run() returned None (no result)"
        )
    return (
        f"not verified: Langroid ended the task as StatusCode.{obs.status}, not as a "
        "finished task; the partial output is not treated as an answer"
    )


def build_suite() -> Dict[str, Any]:
    return {
        "$schema": "https://evalport.org/schema/suite.json",
        "version": SPEC_VERSION,
        "id": SUITE_ID,
        "name": "Langroid Task terminal states",
        "description": (
            "Eight Langroid Task configurations, each asked the same question. They "
            "differ in how the Task ends, not in the question."
        ),
        "graders": [
            {"id": GRADER_ID, "type": "exact_match", "params": {"ignore_case": True}}
        ],
        "test_cases": [
            {
                "id": sc.id,
                "input": QUESTION,
                "expected_output": EXPECTED,
                "graders": [GRADER_ID],
                "metadata": {
                    "langroid": {
                        "scenario": sc.description,
                        "expected_status": sc.expected_status,
                    }
                },
            }
            for sc in SCENARIOS
        ],
        "metadata": {
            "langroid": {
                "version": langroid_version(),
                "llm": "langroid MockLM (no network, no API key)",
            }
        },
    }


def langroid_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("langroid")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def _grader_result(passed: bool) -> Dict[str, Any]:
    return {
        "grader_id": GRADER_ID,
        "type": "exact_match",
        "score": 1 if passed else 0,
        "passed": passed,
    }


def _result_for(
    obs: Observation, policy: str, proposed_verdict: bool, include_durations: bool
) -> Dict[str, Any]:
    langroid_meta: Dict[str, Any] = {
        "status": obs.status,
        "run_returned_none": obs.run_returned_none,
    }
    meta: Dict[str, Any] = {"langroid": langroid_meta}
    result: Dict[str, Any] = {"test_case_id": obs.scenario.id}
    if obs.content is not None:
        result["actual_output"] = obs.content

    if policy == "naive":
        # What a harness that grades whatever run() returned would write: None becomes "",
        # every run is scored, an exception is an error with a 0.
        graded = exact_match(obs.content or "", EXPECTED)
        gr = _grader_result(graded)
        if obs.exception is not None:
            gr["reason"] = f"Task.run raised {obs.exception}"
        elif not graded:
            gr["reason"] = f"Expected {EXPECTED}"
        result["grader_results"] = [gr]
        result["passed"] = graded
        verdict = "passed" if graded else "failed"
    elif is_judged(obs):
        graded = exact_match(obs.content or "", EXPECTED)
        gr = _grader_result(graded)
        if not graded:
            gr["reason"] = f"Expected {EXPECTED}"
        result["grader_results"] = [gr]
        result["passed"] = graded
        verdict = "passed" if graded else "failed"
    else:
        # SPEC Validation Rule 6: score null = "not verified", passed false; a Result
        # whose graders are all null-scored is passed false and SHOULD say so via
        # metadata.openeval.aggregation_status.
        result["grader_results"] = [
            {
                "grader_id": GRADER_ID,
                "type": "exact_match",
                "score": None,
                "passed": False,
                "reason": unverified_reason(obs),
            }
        ]
        result["passed"] = False
        meta["openeval"] = {"aggregation_status": "unscored"}
        verdict = "unverified"

    # An exception, or Langroid's own TIMEOUT, is also a runner-level error. A bare
    # error row is exactly the kind a consumer may drop from its denominators, which is
    # the concern behind Discussion #49.
    if obs.exception is not None:
        result["error"] = {
            "type": "runner_error",
            "message": obs.exception,
            "retryable": False,
        }
    elif obs.status == "TIMEOUT":
        result["error"] = {
            "type": "timeout",
            "message": "Langroid StatusCode.TIMEOUT: run(max_time) elapsed",
            "retryable": False,
        }

    if proposed_verdict:
        result["verdict"] = verdict
    if include_durations:
        result["duration_ms"] = obs.duration_ms
    result["metadata"] = meta
    return result


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Producer-side summary.

    SPEC does not say how ``summary`` counts a Result whose graders are all unscored, so
    this example makes one explicit choice: they are ``skipped``, so that
    ``total == passed + failed + skipped``; ``pass_rate`` is passed / total (an unverified
    run stays in the denominator and is not a pass); ``avg_score`` and ``by_grader`` skip
    null scores, as Rule 6 requires.
    """
    total = len(results)

    def scored(r: Dict[str, Any]) -> bool:
        return any(g["score"] is not None for g in r["grader_results"])

    passed = sum(1 for r in results if r["passed"])
    skipped = sum(1 for r in results if not scored(r))
    failed = total - passed - skipped
    scores = [g["score"] for r in results for g in r["grader_results"] if g["score"] is not None]
    by_passed = sum(1 for r in results for g in r["grader_results"] if g["score"] is not None and g["passed"])
    by_failed = len(scores) - by_passed
    summary: Dict[str, Any] = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "pass_rate": round(passed / total, 4) if total else 0.0,
    }
    if scores:
        avg = round(sum(scores) / len(scores), 4)
        summary["avg_score"] = avg
        summary["by_grader"] = {
            GRADER_ID: {"passed": by_passed, "failed": by_failed, "avg_score": avg}
        }
    return summary


def build_result_set(
    observations: List[Observation],
    *,
    policy: str = "status-aware",
    proposed_verdict: bool = False,
    deterministic: bool = False,
) -> Dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    started = FIXED_STARTED_AT if deterministic else _now()
    results = [
        _result_for(o, policy, proposed_verdict, include_durations=not deterministic)
        for o in observations
    ]
    doc: Dict[str, Any] = {
        "$schema": "https://evalport.org/schema/resultset.json",
        "version": SPEC_VERSION,
        "suite_id": SUITE_ID,
        "run_id": RUN_ID if policy == "status-aware" else f"{RUN_ID}_naive",
        "started_at": started,
        "runner": {"name": "langroid-example", "version": langroid_version()},
        "results": results,
        "summary": summarize(results),
        "metadata": {
            "langroid": {
                "version": langroid_version(),
                "llm": "langroid MockLM (no network, no API key)",
                "policy": policy,
            }
        },
    }
    if not deterministic:
        doc["completed_at"] = _now()
    return doc


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_documents(
    observations: List[Observation],
    *,
    policy: str = "status-aware",
    proposed_verdict: bool = False,
    deterministic: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return build_suite(), build_result_set(
        observations,
        policy=policy,
        proposed_verdict=proposed_verdict,
        deterministic=deterministic,
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _table(observations: List[Observation], results: List[Dict[str, Any]]) -> str:
    rows = [("scenario", "langroid status", "run() returned", "passed", "score")]
    for o, r in zip(observations, results):
        score = r["grader_results"][0]["score"]
        rows.append(
            (
                o.scenario.id,
                o.status or "-",
                "None" if o.run_returned_none else ("raised" if o.exception else repr(o.content)),
                str(r["passed"]).lower(),
                "null" if score is None else str(score),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in rows)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Run Langroid Task scenarios and write EvalPort suite.json and results.json."
    )
    p.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("."))
    p.add_argument("--policy", choices=POLICIES, default="status-aware")
    p.add_argument(
        "--proposed-verdict",
        action="store_true",
        help="add Result.verdict as proposed in EvalPort Discussion #49 (NOT in the spec)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="fixed started_at, no timestamps/durations (for sample_output/ and diffs)",
    )
    args = p.parse_args(argv)

    observations = run_all()
    suite, rs = build_documents(
        observations,
        policy=args.policy,
        proposed_verdict=args.proposed_verdict,
        deterministic=args.deterministic,
    )

    from openeval.validate import validate_result_set, validate_suite

    for name, doc, fn in (("suite", suite, validate_suite), ("results", rs, validate_result_set)):
        v = fn(doc)
        if not v.valid:
            print(f"internal error: generated {name} is invalid: {v.errors}", file=sys.stderr)
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "suite.json").write_text(json.dumps(suite, indent=2) + "\n")
    (args.out_dir / "results.json").write_text(json.dumps(rs, indent=2) + "\n")
    print(_table(observations, rs["results"]))
    s = rs["summary"]
    print(
        f"\npolicy={args.policy}: total={s['total']} passed={s['passed']} failed={s['failed']} "
        f"unverified(skipped)={s['skipped']} pass_rate={s['pass_rate']} avg_score={s.get('avg_score')}"
    )
    print(f"wrote {args.out_dir / 'suite.json'} and {args.out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
