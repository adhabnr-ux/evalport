"""Convert `OpenCompass <https://github.com/open-compass/opencompass>`_
``CustomDataset`` rows and real evaluator scoring output to/from the
EvalPort open evaluation format.

OpenCompass is a large-scale LLM evaluation platform (100+ built-in
benchmark datasets, its own inference/prompt-template/evaluator pipeline).
Most of its 100+ per-benchmark dataset loaders under ``opencompass/datasets/``
are one-off Python files, each with its own bespoke ``load()`` -- there is
no single portable object every benchmark shares. But OpenCompass *does*
expose one genuinely generic, portable surface: ``CustomDataset``
(``opencompass/datasets/custom.py``), the "bring your own eval data" path
documented at ``docs/en/advanced_guides/custom_dataset.md`` -- a flat list
of dict rows loaded from a ``.jsonl``/``.csv`` file, each row either a
multiple-choice question (single-uppercase-letter columns ``A``, ``B``,
``C``, ... plus an ``output_column`` naming the correct letter) or a
free-text QA pair (an ``output_column`` naming the reference answer). This
is the real, portable "evaluation dataset row" shape OpenCompass itself
uses when a user isn't running one of its 100+ curated benchmarks -- and
it maps directly onto EvalPort's own portable ``TestCase`` (input +
expected_output + graders), which is exactly the interchange problem
EvalPort exists to solve.

Everything here was verified directly against the real, installed
``opencompass`` package (0.5.3, the current PyPI release as of 2026-08-21),
not against documentation or docstrings alone:

- ``opencompass.datasets.custom.CustomDataset.load()`` was called against a
  real ``.jsonl`` file and its output shape (a ``datasets.Dataset`` whose
  rows are plain dicts) confirmed directly.
- ``opencompass.datasets.custom.OptionSimAccEvaluator.score(predictions,
  references, test_set)`` was called with real predictions/references/rows
  and its return shape confirmed to be
  ``{"accuracy": <0-100 float>, "details": {"<row index str>": {"pred",
  "parsed", "refr", "correct"}, ...}}``. Reading
  ``opencompass/tasks/openicl_eval.py`` (the task that actually writes
  OpenCompass's own on-disk per-dataset result JSON files, which
  ``opencompass.summarizers.default.DefaultSummarizer`` reads back) confirms
  this ``details`` dict, when an evaluator provides one natively, is dumped
  into that result file *verbatim* -- so this is not a shape invented by
  this adapter, it is OpenCompass's own real result format when
  ``dump_details=True`` is set on a run.
- ``opencompass.openicl.icl_evaluator.AccEvaluator.score(predictions=...,
  references=...)`` (the evaluator used for free-text QA, via
  ``make_qa_gen_config`` in the same ``custom.py``) was likewise called
  directly. Unlike ``OptionSimAccEvaluator``, its return value is just
  ``{"accuracy": <0-100 float>}`` -- no native per-item ``details``. Reading
  ``AccEvaluator._preprocess`` (``opencompass/openicl/icl_evaluator/
  icl_hf_evaluator.py``) shows *why*: it maps every distinct
  prediction/reference string to an integer id, then hands those ids to
  HuggingFace ``evaluate``'s ``accuracy`` metric -- mathematically
  identical to plain elementwise ``str(pred) == str(ref)``, since distinct
  strings never collide onto the same id and equal strings always do. This
  adapter's ``result_to_openeval()`` computes that per-item equality
  directly for the QA path (see the function's docstring for the full
  reasoning), and a test in this package cross-checks that doing so
  reproduces ``AccEvaluator``'s own real aggregate ``accuracy`` exactly.

What this adapter deliberately does NOT attempt: reproducing OpenCompass's
own default prompt-rendering template (``make_mcq_gen_config`` /
``make_qa_gen_config`` in ``custom.py`` build a specific ``HUMAN``/``BOT``
turn template from a row) or its 100+ curated, framework-specific benchmark
loaders (``gsm8k.py``, ``mmlu.py``, ...). Prompt formatting is a runner-side
concern, not a data-interchange concern -- EvalPort's ``TestCase.input`` is
the underlying question/context text, not a rendered chat template, and
every other adapter in this ecosystem draws that same line (see e.g.
``dspy-openeval-adapter``'s explanation of why the metric function itself
isn't serialized). And each curated benchmark loader has its own bespoke
row schema with no shared structure to generalize over from the outside;
``CustomDataset`` is the one surface OpenCompass itself designed to be
generic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "to_openeval",
    "from_openeval",
    "result_to_openeval",
]

_RESERVED_METADATA_KEY = "opencompass"


def _validate_options(options: Sequence[str]) -> List[str]:
    options = list(options)
    bad = [
        o for o in options if not (isinstance(o, str) and len(o) == 1 and o.isupper())
    ]
    if bad:
        raise ValueError(
            "options must each be a single uppercase letter -- this is "
            "OpenCompass's own convention (see "
            "opencompass/datasets/custom.py:OptionSimAccEvaluator.__init__, "
            f"which raises on anything else). Got invalid entries: {bad!r}"
        )
    if not options:
        raise ValueError("options must be a non-empty list of single uppercase letters")
    return options


def _flatten(row: Mapping[str, Any], columns: Sequence[str]) -> str:
    lines = [f"{k}: {row[k]}" for k in columns if row.get(k) is not None]
    if not lines:
        raise ValueError(
            f"none of the requested input columns {list(columns)!r} had a "
            f"non-null value in row: {dict(row)!r}"
        )
    return "\n".join(lines)


def _default_input_columns(
    row: Mapping[str, Any], options: Optional[Sequence[str]], output_column: str
) -> List[str]:
    # Every column except output_column becomes part of the input by
    # default -- including the option columns themselves (A/B/C/D and their
    # choice text) for an MCQ row. Excluding them would silently produce an
    # unanswerable question (the model would see "What is the capital of
    # France?" with no choices to pick from), so this is not optional.
    return [k for k in row.keys() if k != output_column]


def _mcq_grader(options: Sequence[str]) -> Dict[str, Any]:
    # OpenCompass's own OptionSimAccEvaluator does fuzzy option parsing
    # (exact letter match, then a regex-based "first option mentioned"
    # extraction, then substring matching against each option's full text,
    # then a Levenshtein-distance fallback -- see
    # OptionSimAccEvaluator.match_any_label in opencompass/datasets/custom.py)
    # before comparing to the reference letter. That is NOT what EvalPort's
    # own exact_match grader type means (a literal string comparison of
    # actual_output to expected_output), so mapping this to exact_match would
    # overclaim precision this adapter doesn't have. custom, with the real
    # evaluator class name as the handler, is the honest choice -- the same
    # convention every adapter in this ecosystem uses for scoring logic with
    # no exact EvalPort-native equivalent (see e.g.
    # azure-ai-evaluation-openeval-adapter's grader-mapping section).
    return {
        "id": "opencompass_option_sim_acc",
        "type": "custom",
        "params": {
            "handler": "opencompass:OptionSimAccEvaluator",
            "options": list(options),
        },
        "description": (
            "Fuzzy-matches the model's free-text completion to one of the "
            "given options (exact letter, then regex extraction, then "
            "substring match against option text, then Levenshtein-distance "
            "fallback) via OpenCompass's real OptionSimAccEvaluator, and "
            "compares the parsed letter to expected_output."
        ),
    }


def _qa_grader() -> Dict[str, Any]:
    # AccEvaluator's real scoring (see the module docstring) is exact string
    # equality after str() coercion -- this genuinely is EvalPort's
    # exact_match, not an approximation of it.
    return {"id": "opencompass_acc", "type": "exact_match"}


def to_openeval(
    rows: Sequence[Mapping[str, Any]],
    *,
    options: Optional[Sequence[str]] = None,
    input_columns: Optional[Sequence[str]] = None,
    output_column: str = "answer",
    suite_id: Optional[str] = None,
    description: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Convert ``CustomDataset``-shaped rows into an EvalPort ``Suite``.

    ``rows`` is anything iterable of dict-like rows -- a plain
    ``list[dict]`` (e.g. loaded straight from a ``.jsonl``/``.csv`` file the
    way ``CustomDataset.load()`` itself does), or the real
    ``datasets.Dataset`` object ``CustomDataset.load()`` returns (iterating
    a ``Dataset`` yields one dict per row, confirmed directly against the
    installed ``datasets`` package this pulls in as a transitive dependency
    of ``opencompass``).

    Pass ``options`` (e.g. ``["A", "B", "C", "D"]``) for a multiple-choice
    dataset -- matching OpenCompass's own ``OptionSimAccEvaluator``
    convention of single-uppercase-letter option columns holding the choice
    text, with ``output_column`` (default ``"answer"``) naming the column
    holding the correct letter. Omit ``options`` for a free-text QA dataset,
    where ``output_column`` names the column holding the reference answer.

    ``input_columns`` controls which columns are flattened into
    ``TestCase.input`` as ``"key: value"`` lines (one per line, in the
    row's own column order) -- by default, every column except
    ``output_column``, which for an MCQ row includes the option columns
    themselves (a question with no visible choices would be unanswerable).
    This does not reproduce OpenCompass's own default prompt-rendering
    template (see the module docstring); it preserves the underlying data
    losslessly instead.

    Every original row is preserved byte-for-byte under
    ``test_case.metadata.opencompass.row``, so ``from_openeval()`` restores
    it exactly on a round trip through this adapter.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("rows is empty -- nothing to convert")
    if options is not None:
        options = _validate_options(options)
    if ids is not None:
        ids = list(ids)
        if len(ids) != len(rows):
            raise ValueError(f"ids has {len(ids)} entries but rows has {len(rows)}")

    test_cases: List[Dict[str, Any]] = []
    for i, raw_row in enumerate(rows):
        row = dict(raw_row)
        if output_column not in row:
            raise ValueError(
                f"row {i} has no {output_column!r} column (columns present: "
                f"{sorted(row.keys())!r}) -- pass the real output_column name"
            )
        if options is not None:
            missing = [opt for opt in options if opt not in row]
            if missing:
                raise ValueError(
                    f"row {i} is missing option column(s) {missing!r} -- every "
                    f"row must have a value for every entry in options={options!r}"
                )

        cols = list(input_columns) if input_columns is not None else _default_input_columns(
            row, options, output_column
        )

        test_case: Dict[str, Any] = {
            "id": ids[i] if ids is not None else f"row_{i}",
            "input": _flatten(row, cols),
            "expected_output": str(row[output_column]),
            "graders": [_mcq_grader(options)] if options is not None else [_qa_grader()],
            "metadata": {
                _RESERVED_METADATA_KEY: {
                    "row": row,
                    "input_columns": cols,
                    "output_column": output_column,
                    "options": list(options) if options is not None else None,
                }
            },
        }
        test_cases.append(test_case)

    suite: Dict[str, Any] = {
        "version": "1.0.0",
        "id": suite_id or "opencompass_custom_dataset",
        "test_cases": test_cases,
    }
    if description is not None:
        suite["description"] = description
    return suite


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reverse ``to_openeval()``: recover ``CustomDataset``-ready row dicts
    from an EvalPort ``Suite``.

    For a suite this adapter produced, every original row (option columns,
    output column, and anything else it had) is restored exactly from
    ``metadata.opencompass.row``. For a suite built elsewhere (no prior
    round trip through this adapter), a plain heuristic fallback is used --
    ``{"question": test_case["input"], "answer":
    test_case["expected_output"]}`` -- since there's no reliable way to
    recover OpenCompass-specific structure (which columns were MCQ options,
    the real output column name) from an arbitrary third-party TestCase.
    The returned list is ready to hand to ``datasets.Dataset.from_list()``
    and feed straight into ``CustomDataset``/``OptionSimAccEvaluator``, or
    to write out as ``.jsonl`` for OpenCompass's own ``CustomDataset.load()``
    to read back.
    """
    rows: List[Dict[str, Any]] = []
    for test_case in suite.get("test_cases", []):
        metadata = test_case.get("metadata") or {}
        info = metadata.get(_RESERVED_METADATA_KEY)
        if info and "row" in info:
            rows.append(dict(info["row"]))
        else:
            rows.append(
                {"question": test_case.get("input"), "answer": test_case.get("expected_output")}
            )
    return rows


def result_to_openeval(
    rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[str],
    *,
    options: Optional[Sequence[str]] = None,
    output_column: str = "answer",
    suite_id: str,
    run_id: str,
    started_at: str,
    completed_at: Optional[str] = None,
    ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Score real model ``predictions`` against ``rows`` using OpenCompass's
    real evaluators, and build an EvalPort ``ResultSet``.

    ``rows`` must be the same rows (and in the same order) passed to
    ``to_openeval()``, so ``test_case_id``s line up
    (``ids[i]``/``f"row_{i}"`` by default, matching ``to_openeval()``'s own
    default). ``predictions`` is the real, already-generated model
    completion string for each row -- this function does not call any LLM
    itself.

    With ``options`` given, this calls the real, installed
    ``opencompass.datasets.custom.OptionSimAccEvaluator(options).score(
    predictions, references, rows)`` directly and reads its real per-item
    ``details`` dict (``{"pred", "parsed", "refr", "correct"}`` per row --
    see the module docstring for how this was confirmed to match
    OpenCompass's own on-disk result-file format). One ``GraderResult`` of
    type ``custom`` per row, ``score`` 1.0/0.0 from the real ``correct``
    flag, with the evaluator's own ``pred``/``parsed``/``refr`` preserved in
    ``metadata``.

    Without ``options`` (QA path), this calls the real, installed
    ``opencompass.openicl.icl_evaluator.AccEvaluator().score(predictions=...,
    references=...)`` for its real aggregate accuracy (preserved under
    ``result_set["metadata"]["opencompass"]["aggregate_accuracy"]`` for
    cross-checking), and separately computes ``str(pred) == str(ref)`` per
    row to build the per-item ``GraderResult``s EvalPort's ``ResultSet``
    schema requires (``results`` is ``minItems: 1``, one per test case) --
    AccEvaluator's own ``.score()`` does not return a per-item breakdown.
    This is not an approximation: as explained in the module docstring,
    AccEvaluator's aggregate accuracy is computed by mapping distinct
    prediction/reference strings to integer ids and scoring those, which is
    mathematically identical to elementwise string equality. This package's
    tests cross-check that ``sum(correct) / len(rows)`` reproduces
    AccEvaluator's own real ``accuracy`` output (as a fraction) exactly, for
    every case exercised -- not just that the shapes match.
    """
    rows = list(rows)
    predictions = list(predictions)
    if not rows:
        raise ValueError("rows is empty -- nothing to convert")
    if len(predictions) != len(rows):
        raise ValueError(f"predictions has {len(predictions)} entries but rows has {len(rows)}")
    if ids is not None:
        ids = list(ids)
        if len(ids) != len(rows):
            raise ValueError(f"ids has {len(ids)} entries but rows has {len(rows)}")

    for i, row in enumerate(rows):
        if output_column not in row:
            raise ValueError(f"row {i} has no {output_column!r} column")
    references = [str(row[output_column]) for row in rows]

    per_item: List[Dict[str, Any]]
    if options is not None:
        options = _validate_options(options)
        for i, row in enumerate(rows):
            missing = [opt for opt in options if opt not in row]
            if missing:
                raise ValueError(f"row {i} is missing option column(s) {missing!r}")
        try:
            from opencompass.datasets.custom import OptionSimAccEvaluator
        except ImportError as e:  # pragma: no cover - exercised only when opencompass absent
            raise ImportError(
                "result_to_openeval(options=...) calls the real, installed "
                "opencompass.datasets.custom.OptionSimAccEvaluator directly "
                "-- install with `pip install opencompass` (or this "
                "package's `[opencompass]` extra) to score MCQ predictions."
            ) from e
        evaluator = OptionSimAccEvaluator(options=list(options))
        scored = evaluator.score(predictions, references, rows)
        aggregate_accuracy = scored["accuracy"]
        details = scored["details"]
        per_item = []
        for i in range(len(rows)):
            d = details[str(i)]
            per_item.append(
                {
                    "correct": bool(d["correct"]),
                    "metadata": {"pred": d["pred"], "parsed": d["parsed"], "refr": d["refr"]},
                }
            )
        grader_id, grader_type = "opencompass_option_sim_acc", "custom"
        evaluator_name = "OptionSimAccEvaluator"
    else:
        try:
            from opencompass.openicl.icl_evaluator import AccEvaluator
        except ImportError as e:  # pragma: no cover - exercised only when opencompass absent
            raise ImportError(
                "result_to_openeval() calls the real, installed "
                "opencompass.openicl.icl_evaluator.AccEvaluator directly -- "
                "install with `pip install opencompass` (or this package's "
                "`[opencompass]` extra) to score predictions."
            ) from e
        evaluator = AccEvaluator()
        scored = evaluator.score(predictions=predictions, references=references)
        aggregate_accuracy = scored["accuracy"]
        per_item = [
            {
                "correct": str(pred) == ref,
                "metadata": {"pred": pred, "refr": ref},
            }
            for pred, ref in zip(predictions, references)
        ]
        grader_id, grader_type = "opencompass_acc", "exact_match"
        evaluator_name = "AccEvaluator"

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(per_item):
        score = 1.0 if item["correct"] else 0.0
        results.append(
            {
                "test_case_id": ids[i] if ids is not None else f"row_{i}",
                "actual_output": predictions[i],
                "grader_results": [
                    {
                        "grader_id": grader_id,
                        "type": grader_type,
                        "score": score,
                        "passed": item["correct"],
                        "metadata": {_RESERVED_METADATA_KEY: item["metadata"]},
                    }
                ],
                "passed": item["correct"],
            }
        )

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    result_set: Dict[str, Any] = {
        "version": "1.0.0",
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
        },
        "metadata": {
            _RESERVED_METADATA_KEY: {
                "aggregate_accuracy": aggregate_accuracy,
                "evaluator": evaluator_name,
            }
        },
    }
    if completed_at is not None:
        result_set["completed_at"] = completed_at
    return result_set
