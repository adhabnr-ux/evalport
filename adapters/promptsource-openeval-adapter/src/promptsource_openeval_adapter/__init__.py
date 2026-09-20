"""Convert promptsource (https://github.com/bigscience-workshop/promptsource)
applied P3 prompt templates to EvalPort (https://github.com/adhabnr-ux/evalport)
suites.

Grounded in promptsource's real ``promptsource/templates.py`` source, not
guessed:

- ``Template.apply(example)`` renders the template's Jinja against one HF
  ``datasets`` example and returns ``[prompt_string, output_string]`` --
  the two-string ``|||``-split contract promptsource's own CONTRIBUTING.md
  documents ("a template must produce two strings: an input and a target").
- ``Template.get_answer_choices_list(example)`` returns a list of rendered
  answer-choice strings for a classification-style template, or ``None``
  for an open-ended one.
- ``Template.metadata`` is a ``Template.Metadata`` object whose ``.metrics``
  is a list drawn from promptsource's own fixed ``MMETRICS`` vocabulary
  (``Accuracy``, ``BLEU``, ``ROUGE``, ``Squad``, ``Trivia QA``, ``Pearson
  Correlation``, ``Spearman Correlation``, ``MultiRC``, ``AUC``, ``COQA
  F1``, ``Edit Distance``, ``Mean Reciprocal Rank``, ``Other``) -- or
  ``None`` when unset.

promptsource's own PyPI release only supports Python 3.7-3.9 (it hasn't
been re-released for newer interpreters), so it can't be installed
alongside a modern ``evalport-sdk`` toolchain in one environment -- this
module therefore duck-types against the interface above (``.apply()``,
``.get_answer_choices_list()``, ``.metadata.metrics``, ``.get_name()``)
rather than importing ``promptsource`` directly, the same pattern already
used by ``autogen-openeval-adapter`` for a framework it can't/won't import.
This adapter's test suite exercises that exact interface using the real,
unmodified Jinja template string and answer-choices expression from
promptsource's own ``ag_news/templates.yaml`` (quoted verbatim in
bigscience-workshop/promptsource#860) rendered through real Jinja2 --
genuine template rendering, not a hand-typed fixture pretending to be one.

Proposed and confirmed with promptsource's maintainer first:
https://github.com/bigscience-workshop/promptsource/issues/860 ("no changes
to PromptSource are necessary if the adapter is in the EvalPort repo").
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "template_to_openeval",
    "dataset_templates_to_openeval",
    "from_openeval",
    "__version__",
]
__version__ = "0.1.0"

# promptsource's own fixed metric vocabulary (promptsource/templates.py's
# module-level `METRICS` set) that maps cleanly onto an EvalPort standard
# grader type with zero fabricated required params. Only "Accuracy" has an
# honest zero-param EvalPort mapping (exact_match) -- see the module
# docstring's precedent in haystack-openeval-adapter's `_infer_grader_type`
# for why every other metric name (BLEU, ROUGE, Pearson Correlation, ...)
# falls back to "custom" rather than guessing at semantic_similarity's
# required `threshold` or llm_judge's required `model`/`prompt`.
_EXACT_MATCH_METRICS = {"Accuracy"}


def _infer_grader(metric_name: str) -> Dict[str, Any]:
    if metric_name in _EXACT_MATCH_METRICS:
        return {
            "id": metric_name,
            "type": "exact_match",
            "description": (
                f"promptsource template metadata.metrics entry '{metric_name}', "
                "inferred as EvalPort's 'exact_match' grader (no additional "
                "params required)."
            ),
        }
    return {
        "id": metric_name,
        "type": "custom",
        "params": {"handler": metric_name},
        "description": (
            f"Placeholder for promptsource's '{metric_name}' metric -- "
            "EvalPort's typed grader options (llm_judge, semantic_similarity, "
            "...) each require real configuration (a model+prompt, a "
            "similarity threshold) this module has no honest value for. Run "
            "the actual metric computation and supply a Grader with your "
            "real configuration if you want one of those types instead."
        ),
    }


def _template_metrics(template: Any) -> List[str]:
    metrics = getattr(template.metadata, "metrics", None)
    return list(metrics) if metrics else []


def template_to_openeval(
    template: Any,
    examples: Sequence[Dict[str, Any]],
    dataset_name: str,
    subset_name: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
    suite_id: Optional[str] = None,
    version: str = OPENEVAL_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply one promptsource ``Template`` across a sequence of HF ``datasets``
    examples and build an EvalPort suite from the results.

    Args:
        template: A promptsource ``Template``-shaped object (duck-typed:
            must expose ``.apply(example)`` -> ``[prompt, target]``,
            ``.get_answer_choices_list(example)`` -> ``list[str] | None``,
            ``.metadata.metrics`` -> ``list[str] | None``, and
            ``.get_name()`` or a ``.name`` attribute).
        examples: A sequence of HF ``datasets``-style example dicts (what
            ``template.apply()`` expects) -- e.g. one dataset split, or a
            slice of one.
        dataset_name: The underlying HF dataset name (e.g. ``"ag_news"``) --
            used to build default test case ids and suite id, and recorded
            in metadata for traceability back to the source dataset.
        subset_name: The dataset's subset/config name, if any.
        ids: Optional explicit test case ids, one per example. Defaults to
            ``f"{dataset_name}[_{subset_name}]_{template_name}_{i}"``.
        suite_id: EvalPort ``Suite.id``; defaults to
            ``f"promptsource_{dataset_name}[_{subset_name}]_{template_name}"``.
        version, description: EvalPort Suite-level fields.

    Returns:
        A dict matching EvalPort's Suite schema (validate with
        ``openeval.validate.validate_suite``).

    Raises:
        ValueError: if ``examples`` is empty, ``ids`` has a mismatched
            length, or ``template.apply()`` doesn't return exactly two
            strings for some example (promptsource's own documented
            contract -- see the module docstring).
    """
    if not examples:
        raise ValueError("template_to_openeval: examples is empty -- nothing to convert.")
    if ids is not None and len(ids) != len(examples):
        raise ValueError(
            f"template_to_openeval: ids has length {len(ids)}, expected "
            f"{len(examples)} (one entry per example)."
        )

    template_name = getattr(template, "name", None) or (
        template.get_name() if hasattr(template, "get_name") else "template"
    )
    metrics = _template_metrics(template)
    grader_dicts = [_infer_grader(m) for m in metrics]
    grader_names = [g["id"] for g in grader_dicts]

    name_parts = [dataset_name] + ([subset_name] if subset_name else [])
    base_name = "_".join(name_parts)
    default_suite_id = f"promptsource_{base_name}_{template_name}"

    test_cases: List[Dict[str, Any]] = []
    for i, example in enumerate(examples):
        rendered = template.apply(example)
        if not isinstance(rendered, (list, tuple)) or len(rendered) != 2:
            raise ValueError(
                f"template_to_openeval: template.apply() must return exactly "
                f"2 strings (promptsource's own '|||'-split contract), got "
                f"{rendered!r} for example index {i}."
            )
        prompt_str, target_str = rendered
        answer_choices = template.get_answer_choices_list(example)

        tc_id = ids[i] if ids else f"{base_name}_{template_name}_{i}"
        test_case: Dict[str, Any] = {
            "id": tc_id,
            "input": prompt_str,
            "expected_output": target_str,
            "graders": list(grader_names) if grader_names else ["exact_match"],
            "metadata": {
                "promptsource": {
                    "dataset": dataset_name,
                    "subset": subset_name,
                    "template_name": template_name,
                }
            },
        }
        if answer_choices is not None:
            test_case["metadata"]["promptsource"]["answer_choices"] = answer_choices
        test_cases.append(test_case)

    if not grader_dicts:
        # No metadata.metrics on this template -- fall back to a plain
        # exact_match grader (no required params, so nothing is fabricated)
        # rather than leaving test_cases.graders dangling with no matching
        # Suite.graders entry, which validate_suite() would flag as
        # DANGLING_REFERENCE.
        grader_dicts = [{"id": "exact_match", "type": "exact_match"}]

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or default_suite_id,
        "graders": grader_dicts,
        "test_cases": test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def dataset_templates_to_openeval(
    dataset_templates: Any,
    examples: Sequence[Dict[str, Any]],
    dataset_name: str,
    subset_name: Optional[str] = None,
    template_names: Optional[Iterable[str]] = None,
    suite_id: Optional[str] = None,
    version: str = OPENEVAL_VERSION,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply every (or a chosen subset of) template in a promptsource
    ``DatasetTemplates`` collection across the same set of examples, merging
    the results into one EvalPort suite -- the exact operation
    bigscience-workshop/promptsource#860 proposed: "Applying a
    DatasetTemplates collection to its underlying HF dataset split would
    produce a ready-made EvalPort Suite."

    Args:
        dataset_templates: A promptsource ``DatasetTemplates``-shaped
            object (duck-typed: must expose ``.all_template_names`` ->
            ``list[str]`` and ``__getitem__(name)`` -> a ``Template``).
        examples: Passed through to ``template_to_openeval`` for every
            selected template.
        dataset_name, subset_name: Passed through to
            ``template_to_openeval``.
        template_names: Which templates to apply, by name. Defaults to
            every template in ``dataset_templates.all_template_names``.
        suite_id, version, description: EvalPort Suite-level fields for the
            merged suite. When multiple templates contribute graders with
            the same ``id`` (e.g. two templates both tagged ``Accuracy``),
            the grader is deduplicated rather than added twice (would
            otherwise trip ``validate_suite()``'s ``DUPLICATE_ID`` check).

    Returns:
        A dict matching EvalPort's Suite schema, with one test case per
        (template, example) pair -- ``len(templates) * len(examples)``
        total.

    Raises:
        ValueError: if ``template_names`` names a template not present in
            ``dataset_templates``, or (propagated from
            ``template_to_openeval``) if ``examples`` is empty.
    """
    names = list(template_names) if template_names is not None else list(
        dataset_templates.all_template_names
    )
    if not names:
        raise ValueError(
            "dataset_templates_to_openeval: no templates to apply (empty "
            "template_names and dataset_templates has none)."
        )

    all_test_cases: List[Dict[str, Any]] = []
    graders_by_id: Dict[str, Dict[str, Any]] = {}

    for name in names:
        try:
            template = dataset_templates[name]
        except KeyError as e:
            raise ValueError(
                f"dataset_templates_to_openeval: template {name!r} not found "
                f"in dataset_templates."
            ) from e

        sub_suite = template_to_openeval(
            template,
            examples,
            dataset_name=dataset_name,
            subset_name=subset_name,
        )
        all_test_cases.extend(sub_suite["test_cases"])
        for g in sub_suite["graders"]:
            graders_by_id.setdefault(g["id"], g)

    name_parts = [dataset_name] + ([subset_name] if subset_name else [])
    base_name = "_".join(name_parts)

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id or f"promptsource_{base_name}",
        "graders": list(graders_by_id.values()),
        "test_cases": all_test_cases,
    }
    if description:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite back into a list of promptsource-shaped
    "applied template" records.

    A live promptsource ``Template`` can't be reconstructed in general (its
    Jinja source and reference metadata aren't part of an applied prompt's
    output), so this returns the flattened record shape this adapter itself
    produces -- ``{input, target, answer_choices, dataset, subset,
    template_name}`` per test case -- mirroring the "reconstruct records,
    not live framework objects" convention ``haystack-openeval-adapter``'s
    ``from_openeval()`` already uses for the same reason.

    Args:
        suite: An EvalPort suite dict.

    Returns:
        A list of plain dicts, one per test case, in suite order.

    Raises:
        ValueError: if the suite has no test cases.
    """
    test_cases = suite.get("test_cases") or []
    if not test_cases:
        raise ValueError("from_openeval: suite has no test_cases to convert.")

    records: List[Dict[str, Any]] = []
    for tc in test_cases:
        ps_meta = ((tc.get("metadata") or {}).get("promptsource")) or {}
        records.append(
            {
                "id": tc.get("id"),
                "input": tc.get("input"),
                "target": tc.get("expected_output"),
                "answer_choices": ps_meta.get("answer_choices"),
                "dataset": ps_meta.get("dataset"),
                "subset": ps_meta.get("subset"),
                "template_name": ps_meta.get("template_name"),
            }
        )
    return records
