"""Convert between Guardrails AI (https://github.com/guardrails-ai/guardrails)
``Guard`` validation runs and EvalPort (https://github.com/adhabnr-ux/evalport)
suites and result sets.

EvalPort is an open interchange format (Apache 2.0) for portable LLM
evaluation datasets: test cases, graders, suites, and results as plain JSON,
shared across evaluation tools (DeepEval, Promptfoo, Inspect AI, AutoGen,
CrewAI, Ragas, LangSmith, Braintrust, MLflow, Opik, Arize Phoenix, Weights &
Biases Weave, UpTrain, Langfuse, Giskard, LlamaIndex, Patronus AI, Vertex AI,
DSPy, Haystack, Evidently, and now Guardrails AI).

Guardrails' native evaluation surface is a ``guardrails.Guard`` with one or
more ``Validator``\\ s attached via ``Guard().use(*validators)``, each
checking a candidate string and returning a ``PassResult`` or ``FailResult``.
There's no separate "input"/"expected output" concept the way a QA or RAG
framework has -- a Guard validates the candidate text itself. This module has
three entry points that mirror that shape directly:

    to_openeval(guard, values, ...)
        Converts a Guard's attached validators plus a list of candidate
        strings into an EvalPort suite (test cases only; no results yet).
        Every attached validator becomes one EvalPort grader; every string
        becomes one test case, graded by all of them (the same semantics as
        calling ``guard.validate()`` -- every attached validator runs on
        every input).

    from_openeval(suite)
        Converts an EvalPort suite back into a list of
        ``{"id": ..., "value": ...}`` dicts, ready to feed one at a time to
        ``guard.validate(item["value"])``.

    evaluation_result_to_openeval(guard, outcomes, ids, ...)
        Converts a list of ``ValidationOutcome`` objects (the results of
        calling ``guard.validate(value)`` for each test case, in order) into
        an EvalPort ResultSet -- one grader result per attached validator,
        per test case.

Why grader ids come from the Guard itself, not a caller-supplied mapping
--------------------------------------------------------------------------

Unlike ``evidently-openeval-adapter`` (whose descriptors have no fixed output
name -- the caller picks an arbitrary ``alias``) or
``haystack-openeval-adapter`` (whose evaluators have a fixed metric name),
a Guardrails ``Validator`` is fully self-describing: ``guard.get_validators()``
returns every attached validator instance, and each one exposes
``.rail_alias`` (its registered name, e.g. ``"guardrails/valid_length"`` for
a Guardrails Hub validator, or whatever name you passed to
``@register_validator`` for a custom one) and ``.get_args()`` (the exact
constructor kwargs it was built with). So this module derives EvalPort
graders directly from the Guard's own validator list -- no caller-supplied
type mapping needed, and nothing guessed.

Why every attached validator maps to EvalPort's ``"custom"`` grader type
--------------------------------------------------------------------------

A Guardrails validator is an arbitrary Python check -- there is no reliable,
generic way to know from the outside whether a given validator class means
"exact string equality" (EvalPort's ``"exact_match"``, the one type with no
required ``params``) versus a length check, a regex, a toxicity classifier,
or a remote-inference call to the Guardrails Hub. Rather than pattern-match
class names against a guess list (which would misclassify the very first
validator whose name doesn't match one of the guesses), every validator maps
to EvalPort's ``"custom"`` grader type, with ``params.handler`` set to the
validator's class name and ``params`` populated with its actual constructor
arguments via ``get_args()`` -- the same "don't fabricate required params"
reasoning documented by every other adapter in this ecosystem.

Why only *failing* validators produce an explicit record, and what this
module does about it
--------------------------------------------------------------------------

This is a real, verified quirk of the Guardrails runtime, not a design
choice of this module: ``ValidationOutcome.to_dict()["validationSummaries"]``
(built by ``ValidationSummary.from_validator_logs_only_fails()`` in
Guardrails' own source) is filtered to failures only -- a validator that
passes produces no entry at all. Confirmed directly against the real
``guardrails-ai`` package (0.11.0): attaching three validators where the
first and second fail and the third passes, all three run (verified via
instrumented ``validate()`` calls), but ``validationSummaries`` contains only
the first two. This module works with that reporting model rather than
against it: for each test case, every validator attached to the guard that
does **not** appear in that outcome's ``validationSummaries`` is recorded as
a pass (``score: 1.0``, ``passed: true``, no ``reason``) -- silence is
success, exactly as Guardrails itself treats it. A validator that does
appear becomes ``score: 0.0``, ``passed: false``, with the real
``failureReason`` preserved as ``grader_result.reason``.

Why a Guard with two instances of the *same* validator class is rejected
--------------------------------------------------------------------------

Also verified directly: when two instances of the same registered validator
(e.g. two separately-configured ``ContainsWord`` checks) are both attached
and both fail, Guardrails' ``validationSummaries`` reports both failures --
but each entry only carries ``validatorName`` (the shared class name) and
``propertyPath`` (always ``"$"`` for a plain-string Guard), never a
per-instance identifier. Naively matching failures back to specific
instances by conservative FIFO order looks tempting but is not sound: which
same-class instances actually passed (and were therefore silently omitted,
per the above) shifts the count on a per-test-case basis, so a purely
positional match can silently attribute a failure to the wrong instance.
Rather than risk that, both ``to_openeval()`` and
``evaluation_result_to_openeval()`` raise ``ValueError`` up front if
``guard.get_validators(on="output")`` contains a duplicate ``rail_alias`` --
the same "don't fabricate, document the real constraint" stance the rest of
this ecosystem takes. Attach each distinct check as its own registered
validator (a very small ``@register_validator`` wrapper if needed) to avoid
this; it's also simply better Guardrails practice, since the Hub's own
validators are registered individually per check for the same reason.

Why ``Guard().use(a).use(b)`` is a footgun this module works around
--------------------------------------------------------------------------

Also verified directly against 0.11.0: chained ``.use()`` calls do **not**
accumulate validators -- each call *replaces* whatever was attached before.
``Guard().use(a).use(b)`` ends up with only ``b`` attached, silently. The
correct way to attach multiple validators is a single call,
``Guard().use(a, b)`` (or ``Guard().use(validators=[a, b])``). This module
doesn't attach validators for you (you build and configure the ``Guard``
yourself, exactly as you would without EvalPort involved), but the README's
usage example is deliberately written the correct way, and this docstring
exists so nobody loses an afternoon to it the way building this adapter's
tests did.

Why ``TestCase.input`` is the candidate string, not a query/response pair
--------------------------------------------------------------------------

A Guardrails ``Validator`` checks one string -- there's no separate "prompt"
and "completion" the way a QA framework has. So ``to_openeval()`` maps each
candidate string directly onto ``TestCase.input`` (a plain string, per
EvalPort's schema), and ``expected_output`` is never populated by this
module -- there is nothing in a Guardrails validation run that corresponds
to it. If your pipeline has a genuine expected/reference value you want
preserved for other purposes, put it in the free-form ``metadata`` on the
test case; ``to_openeval()`` never touches that field for entries under any
key other than ``metadata.guardrails``.

What round-trips losslessly, and what doesn't
-----------------------------------------------

Guardrails -> EvalPort -> Guardrails (via this adapter's ``from_openeval()``):
lossless for the candidate strings and their ids. The suite's ``graders``
carry each validator's class name and constructor args (via ``get_args()``),
which is everything needed to *describe* the check, but this module does not
reconstruct live ``Validator`` instances from a suite -- the caller
re-attaches the real validators to a ``Guard`` themselves, the same way
every framework adapter in this ecosystem treats "the grader's actual
implementation" as something the framework runs, not something EvalPort
re-executes.

Guardrails -> EvalPort -> some other tool: the candidate strings and pass/
fail/score/reason per validator are readable by any EvalPort consumer, but a
different tool has no way to *run* a Guardrails-specific validator -- the
same tradeoff every adapter in this ecosystem takes for framework-specific
logic with no native EvalPort equivalent.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

try:
    import guardrails  # noqa: F401  (import-time dependency check only)
except ImportError as e:  # pragma: no cover - exercised by the packaging itself
    raise ImportError(
        "guardrails-openeval-adapter requires the 'guardrails-ai' package. "
        "Install it with: pip install guardrails-ai"
    ) from e

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk not installed
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "to_openeval",
    "from_openeval",
    "evaluation_result_to_openeval",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_native(value: Any) -> Any:
    """Best-effort conversion of a validator constructor argument into a
    plain JSON-serializable Python value. Guardrails validator kwargs are
    ordinarily plain Python primitives (str/int/float/bool/list/dict/None),
    but this defends against an oddball object slipping through rather than
    breaking suite construction outright."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_native(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_native(v) for k, v in value.items()}
    return str(value)


def _get_output_validators(guard: Any) -> List[Any]:
    """Return the Validator instances attached to ``guard`` for the
    ``"output"`` property -- what ``guard.validate(value)`` actually runs
    against a plain string. Raises ValueError if none are attached, or if
    two attached validators share the same ``rail_alias`` (see the module
    docstring's "Why a Guard with two instances of the same validator class
    is rejected" section)."""
    validators = list(guard.get_validators(on="output"))
    if not validators:
        raise ValueError(
            "guard has no validators attached for 'output' -- nothing to "
            "build EvalPort graders from. Attach validators with "
            "Guard().use(validator_a, validator_b, ...) (a single call -- "
            "see the module docstring's '.use(a).use(b)' warning) before "
            "converting."
        )
    aliases = [v.rail_alias for v in validators]
    seen: Dict[str, int] = {}
    duplicates = set()
    for alias in aliases:
        seen[alias] = seen.get(alias, 0) + 1
        if seen[alias] > 1:
            duplicates.add(alias)
    if duplicates:
        raise ValueError(
            "guard has more than one validator registered as "
            f"{sorted(duplicates)!r}. This adapter cannot reliably attribute "
            "a validationSummaries failure back to one specific instance "
            "when the same validator is attached more than once (see the "
            "module docstring's 'Why a Guard with two instances of the same "
            "validator class is rejected' section). Register each distinct "
            "check as its own validator to disambiguate."
        )
    return validators


def _grader_dict_for_validator(validator: Any) -> Dict[str, Any]:
    class_name = type(validator).__name__
    args = {k: _to_native(v) for k, v in (validator.get_args() or {}).items()}
    on_fail = getattr(validator, "on_fail_descriptor", None)
    params: Dict[str, Any] = {"handler": class_name, **args}
    if on_fail is not None:
        params["on_fail"] = str(getattr(on_fail, "value", on_fail))
    return {
        "id": validator.rail_alias,
        "type": "custom",
        "params": params,
        "description": (
            f"Guardrails validator '{class_name}' (registered as "
            f"'{validator.rail_alias}') -- the caller must run the actual "
            "guard.validate() call and supply its outcome via "
            "evaluation_result_to_openeval(), rather than this module "
            "fabricating a fake implementation."
        ),
    }


def to_openeval(
    guard: Any,
    values: Sequence[str],
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    version: str = OPENEVAL_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an EvalPort suite from a configured Guard and a list of
    candidate strings to validate.

    Args:
        guard: A ``guardrails.Guard`` with validators already attached via
            ``Guard().use(validator_a, validator_b, ...)`` (a single call --
            see the module docstring's ``.use(a).use(b)`` warning). Every
            attached ``"output"`` validator becomes one EvalPort grader.
        values: The candidate strings to validate -- what you would pass to
            ``guard.validate(value)`` one at a time. Each becomes one
            ``TestCase`` whose ``input`` is that string, graded by every
            validator attached to ``guard`` (matching the real semantics of
            ``guard.validate()``: every attached validator runs on every
            call).
        ids: Optional explicit test case ids; auto-generated
            (``guardrails_tc_<n>``) if omitted.
        suite_id: EvalPort ``Suite.id``; defaults to ``"guardrails_suite"``.
        version, description: EvalPort Suite-level fields.

    Returns:
        A dict matching EvalPort's Suite schema
        (validate with ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``values`` is empty, ``guard`` has no validators
            attached, two attached validators share a ``rail_alias`` (see
            the module docstring), or ``ids`` has a mismatched length.
    """
    if not values:
        raise ValueError("to_openeval: values is empty -- nothing to convert.")
    if ids is not None and len(ids) != len(values):
        raise ValueError(
            f"to_openeval: ids has length {len(ids)}, expected {len(values)} "
            "(one entry per value)."
        )

    validators = _get_output_validators(guard)
    grader_dicts = [_grader_dict_for_validator(v) for v in validators]
    grader_ids = [g["id"] for g in grader_dicts]

    test_cases: List[Dict[str, Any]] = []
    for i, value in enumerate(values):
        tc_id = ids[i] if ids else f"guardrails_tc_{i}"
        test_cases.append(
            {
                "id": tc_id,
                "input": str(value),
                "graders": list(grader_ids),
                "metadata": {"guardrails": {"validator_count": len(grader_ids)}},
            }
        )

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or "guardrails_suite",
        "graders": grader_dicts,
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite back into a list of candidate strings ready
    to hand to ``guard.validate()``.

    Each entry is a ``{"id": test_case_id, "value": candidate_string}`` dict,
    in suite order -- the id is returned explicitly (rather than this module
    hiding it, or dropping it) because ``guard.validate()`` returns only a
    ``ValidationOutcome`` with no reference back to which test case produced
    it; the caller needs the id at hand to pass to
    ``evaluation_result_to_openeval()`` afterward. Usage::

        for item in from_openeval(suite):
            outcome = guard.validate(item["value"])
            outcomes.append(outcome)
            ids.append(item["id"])

    A test case whose ``input`` is a plain string is used as-is. A test case
    whose ``input`` is an array of strings (EvalPort's other valid shape,
    used by tools that flatten multiple named fields into one input -- see
    e.g. ``evidently-openeval-adapter``) is joined with ``"\\n"`` into a
    single string, since a Guardrails validator checks exactly one string
    with no concept of multiple named parts.

    Args:
        suite: An EvalPort suite dict.

    Returns:
        A list of ``{"id": ..., "value": ...}`` dicts, one per test case.

    Raises:
        ValueError: if the suite has no test cases.
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    items: List[Dict[str, Any]] = []
    for tc in test_cases:
        raw_input = tc.get("input")
        value = "\n".join(raw_input) if isinstance(raw_input, list) else str(raw_input)
        items.append({"id": tc.get("id"), "value": value})
    return items


def evaluation_result_to_openeval(
    guard: Any,
    outcomes: Sequence[Any],
    ids: Sequence[str],
    suite_id: str = "guardrails_suite",
    run_id: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    version: str = OPENEVAL_VERSION,
) -> Dict[str, Any]:
    """Convert a list of Guard validation outcomes into an EvalPort
    ResultSet.

    Args:
        guard: The same ``guardrails.Guard`` used to produce ``outcomes``
            (source of truth for which validators exist -- see the module
            docstring's "Why grader ids come from the Guard itself" section).
        outcomes: A sequence of ``guardrails.classes.ValidationOutcome``
            objects, i.e. the return value of ``guard.validate(value)`` --
            one per test case, in the same order as ``ids``.
        ids: The test case id for each entry of ``outcomes`` (same length,
            same order). Required: a ``ValidationOutcome`` carries no
            reference back to which EvalPort test case produced it -- see
            ``from_openeval()``'s docstring for how to keep them paired up.
        suite_id: The EvalPort suite this ResultSet's ``test_case_id``s
            refer to.
        run_id: EvalPort ``ResultSet.run_id``; a random one is generated if
            omitted.
        started_at, completed_at: ISO-8601 timestamps. ``started_at``
            defaults to now if omitted.
        version: EvalPort schema version.

    Returns:
        A dict matching EvalPort's ResultSet schema
        (validate with ``openeval.validate.validate_result_set``).

    Raises:
        ValueError: if ``outcomes``/``ids`` is empty, they have mismatched
            lengths, ``guard`` has no validators attached, or two attached
            validators share a ``rail_alias`` (see the module docstring).
    """
    if not outcomes:
        raise ValueError("evaluation_result_to_openeval: outcomes is empty.")
    if len(outcomes) != len(ids):
        raise ValueError(
            f"evaluation_result_to_openeval: outcomes has length "
            f"{len(outcomes)} but ids has length {len(ids)} -- they must "
            "be the same length and in the same order."
        )

    validators = _get_output_validators(guard)
    validator_class_names = [type(v).__name__ for v in validators]
    grader_ids = [v.rail_alias for v in validators]

    results_out: List[Dict[str, Any]] = []
    for tc_id, outcome in zip(ids, outcomes):
        outcome_dict = outcome.to_dict()
        # Guardrails only reports FAILING validators here -- see the module
        # docstring's "Why only failing validators produce an explicit
        # record" section. Every attached validator not named in this list
        # is a pass, per Guardrails' own reporting model.
        failures = outcome_dict.get("validationSummaries") or []
        failure_by_name: Dict[str, str] = {}
        for f in failures:
            failure_by_name[f.get("validatorName")] = f.get("failureReason") or ""

        grader_results: List[Dict[str, Any]] = []
        for grader_id, class_name in zip(grader_ids, validator_class_names):
            if class_name in failure_by_name:
                grader_results.append(
                    {
                        "grader_id": grader_id,
                        "type": "custom",
                        "score": 0.0,
                        "passed": False,
                        "reason": failure_by_name[class_name],
                    }
                )
            else:
                grader_results.append(
                    {
                        "grader_id": grader_id,
                        "type": "custom",
                        "score": 1.0,
                        "passed": True,
                    }
                )

        result_entry: Dict[str, Any] = {
            "test_case_id": str(tc_id),
            "grader_results": grader_results,
            "passed": all(gr["passed"] for gr in grader_results),
        }
        actual_output = outcome_dict.get("validatedOutput")
        if actual_output is None:
            actual_output = outcome_dict.get("rawLlmOutput")
        if actual_output is not None:
            result_entry["actual_output"] = str(actual_output)

        results_out.append(result_entry)

    total = len(results_out)
    passed_count = sum(1 for r in results_out if r["passed"])

    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"guardrails_run_{uuid.uuid4().hex[:12]}",
        "started_at": started_at or _now_iso(),
        "results": results_out,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
        },
    }
    if completed_at:
        result_set["completed_at"] = completed_at
    return result_set
