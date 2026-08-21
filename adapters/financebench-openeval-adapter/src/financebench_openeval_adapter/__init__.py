"""Convert FinanceBench (https://github.com/patronus-ai/financebench) rows and
scored model completions to and from EvalPort (https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation test cases, graders,
suites, and results.

FinanceBench is a static-data benchmark, not a Python package: it ships as two
JSONL files under `data/` (`financebench_open_source.jsonl`, 150 open-book
financial-QA questions with human-annotated gold answers and evidence, and
`financebench_document_information.jsonl`, source-document metadata keyed by
`doc_name`) plus a `results/` directory of JSONL files recording model
completions on those questions with a human-annotated correctness `label`
("Correct Answer", "Incorrect Answer", or "Refusal" -- confirmed by loading a
real results file in this adapter's test suite; no other label values were
observed). There is no pip-installable `financebench` package to depend on --
this adapter works directly against those JSONL row shapes (plain dicts, e.g.
loaded via `json.loads` per line, or via `pandas.read_json(..., lines=True)`
per FinanceBench's own README), so it has no third-party runtime dependency
beyond `evalport-sdk`.

One thing worth flagging: FinanceBench's own README documents the document-info
column as `comany_sector_gics`, but the real file on the `main` branch (verified
by downloading it in this adapter's test setup) actually uses `gics_sector`.
This adapter reads the real field name (`gics_sector`) and falls back to the
documented-but-unobserved `comany_sector_gics` only if present, so it works
whether that README typo ever gets fixed upstream.

## Why `llm_judge`, not `exact_match`, for `to_openeval()`

FinanceBench answers are free-text financial statements ("$1577.00 million",
"Increased by ~15% YoY", short multi-sentence explanations) graded by human
judgment in the original paper and in the `results/` files this adapter also
converts -- not by exact string equality. Grading these honestly requires
either the same human judgment the benchmark's authors used, or an `llm_judge`
grader approximating it; `exact_match` would silently misgrade the large
majority of correct-but-differently-formatted answers ("1,577" vs "$1577.00
million" is the same fact, wrong under `exact_match`). `to_openeval()`
therefore emits an `llm_judge` grader by default, with the evaluation criteria
FinanceBench's own paper describes (numerical/factual correctness against the
gold answer and cited evidence) baked into the prompt -- pick your own
`judge_model` (the params.model default, "gpt-4o", is a placeholder, not a
recommendation) and swap the prompt if your grading criteria differ.

## Why `result_to_openeval()` uses grader type `"human"`, not `"llm_judge"`

The `results/*.jsonl` files already carry a real human-annotated `label` per
row -- this function does not re-judge anything with an LLM, it losslessly
carries that existing human judgment into EvalPort's `Result`/`GraderResult`
shape (`type: "human"`, which -- like `exact_match` -- has no required
`params` per `spec/schemas/grader.json`, so no fabricated parameters are
needed). Fabricating a fresh `llm_judge` score here, when a real human label
already exists in the source data, would throw away information the original
benchmark authors spent manual review effort producing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "to_openeval",
    "from_openeval",
    "result_to_openeval",
    "load_jsonl",
]

_SPEC_VERSION = "1.0.0"
_GRADER_ID = "gr_financebench_judge"
_HUMAN_GRADER_ID = "gr_financebench_human_label"

# Confirmed by loading a real results/*.jsonl file (gpt-4_sharedStore.jsonl) in
# this adapter's test suite: exactly these three label strings occur, no others.
_PASSING_LABELS = {"Correct Answer"}
_KNOWN_LABELS = {"Correct Answer", "Incorrect Answer", "Refusal"}

_DEFAULT_JUDGE_PROMPT = (
    "You are grading a financial question-answering system. "
    "Question: {input}\n"
    "Gold (human-annotated) answer: {expected}\n"
    "Model's answer: {output}\n\n"
    "Judge whether the model's answer is factually and numerically consistent "
    "with the gold answer (minor formatting/rounding differences, e.g. "
    '"$1,577 million" vs "1577.00", are NOT errors; a refusal to answer, a '
    "materially different number, or a claim the evidence does not support IS "
    'an error). Return JSON: {"score": 0.0-1.0, "reasoning": "..."}.'
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a FinanceBench `.jsonl` file (any of `data/financebench_open_source.jsonl`,
    `data/financebench_document_information.jsonl`, or a `results/*.jsonl` file)
    into a list of plain dicts, one per line. A thin convenience wrapper --
    every function in this module also accepts an already-loaded list of dicts
    directly, so this is optional."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sector(doc_row: Dict[str, Any]) -> Optional[str]:
    # Real field on `main` is `gics_sector`; FinanceBench's own README documents
    # (but this adapter never observed on disk) `comany_sector_gics`. Prefer the
    # real one, fall back to the documented one so this keeps working either way.
    return doc_row.get("gics_sector", doc_row.get("comany_sector_gics"))


# Fields that only exist in financebench_document_information.jsonl (joined
# in by to_openeval() via doc_name), as opposed to financebench_open_source.jsonl
# itself -- from_openeval() uses this to reconstruct the right file's row shape.
_DOC_INFO_ONLY_FIELDS = {"doc_type", "doc_period", "doc_link", "gics_sector"}


def _doc_metadata(doc_row: Dict[str, Any]) -> Dict[str, Any]:
    # Deliberately excludes "company": financebench_open_source.jsonl rows
    # already carry their own "company" field (handled by the main field loop
    # in to_openeval()), so joining the document-info file's "company" here
    # too would just overwrite an identical value with itself.
    meta: Dict[str, Any] = {}
    for key in ("doc_type", "doc_period", "doc_link"):
        if key in doc_row:
            meta[f"financebench.{key}"] = doc_row[key]
    sector = _sector(doc_row)
    if sector is not None:
        meta["financebench.gics_sector"] = sector
    return meta


def to_openeval(
    question_rows: Iterable[Dict[str, Any]],
    document_info_rows: Optional[Iterable[Dict[str, Any]]] = None,
    suite_id: str = "financebench_open_source",
    judge_model: str = "gpt-4o",
    judge_prompt: str = _DEFAULT_JUDGE_PROMPT,
) -> Dict[str, Any]:
    """Convert FinanceBench question rows (the shape of
    `data/financebench_open_source.jsonl`: `financebench_id`, `question`,
    `answer`, `evidence`, `company`, `doc_name`, `question_type`,
    `question_reasoning`, `domain_question_num`, `justification`,
    `dataset_subset_label`) into an EvalPort suite (dict).

    `document_info_rows`, if given (the shape of
    `data/financebench_document_information.jsonl`), is joined onto each
    question row by `doc_name` -- matching the join FinanceBench's own README
    documents (`pd.merge(df_questions, df_meta, on="doc_name")`) -- and folded
    into `TestCase.metadata` as `financebench.doc_type` /
    `financebench.doc_period` / `financebench.doc_link` /
    `financebench.gics_sector`. Rows with no matching `doc_name` in
    `document_info_rows` are still converted; they simply get no
    `financebench.doc_*` metadata keys.

    `question_rows` may be an already-loaded list of dicts, or use
    `load_jsonl()` to read directly from a `.jsonl` file. Returns a plain dict
    conforming to the EvalPort EvalSuite schema -- pass it to
    `openeval.validate.validate_suite()` to confirm compliance.
    """
    question_rows = list(question_rows)
    doc_by_name: Dict[str, Dict[str, Any]] = {}
    if document_info_rows is not None:
        for d in document_info_rows:
            name = d.get("doc_name")
            if isinstance(name, str):
                doc_by_name[name] = d

    test_cases: List[Dict[str, Any]] = []
    for row in question_rows:
        fb_id = row.get("financebench_id")
        tc_id = str(fb_id) if fb_id is not None else f"financebench_row_{len(test_cases)}"

        evidence = row.get("evidence") or []
        context = [
            e.get("evidence_text", "")
            for e in evidence
            if isinstance(e, dict) and e.get("evidence_text")
        ]

        metadata: Dict[str, Any] = {"financebench.evidence": evidence}
        for key in (
            "company",
            "doc_name",
            "question_type",
            "question_reasoning",
            "domain_question_num",
            "justification",
            "dataset_subset_label",
        ):
            if key in row:
                metadata[f"financebench.{key}"] = row[key]

        doc_name = row.get("doc_name")
        if isinstance(doc_name, str) and doc_name in doc_by_name:
            metadata.update(_doc_metadata(doc_by_name[doc_name]))

        tc: Dict[str, Any] = {
            "id": tc_id,
            "input": row.get("question", ""),
            "graders": [_GRADER_ID],
            "metadata": metadata,
        }
        if "answer" in row:
            tc["expected_output"] = row["answer"]
        if context:
            tc["context"] = context
        test_cases.append(tc)

    grader = {
        "id": _GRADER_ID,
        "type": "llm_judge",
        "description": (
            "Judges financial-QA correctness against the FinanceBench gold "
            "answer, tolerant of numeric formatting differences."
        ),
        "params": {"model": judge_model, "prompt": judge_prompt},
    }

    return {
        "version": _SPEC_VERSION,
        "id": suite_id,
        "description": (
            "FinanceBench (https://github.com/patronus-ai/financebench): "
            "open-book financial question answering over 10-K/10-Q/8-K/earnings "
            "documents, converted from the open-source 150-example release."
        ),
        "test_cases": test_cases,
        "graders": [grader],
    }


def from_openeval(suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an EvalPort suite back into FinanceBench
    `financebench_open_source.jsonl` row shape (best-effort round trip).

    Any `TestCase.metadata["financebench.*"]` key set by `to_openeval()` is
    unpacked back to its original FinanceBench field name (e.g.
    `financebench.question_type` -> `question_type`); test cases with no such
    metadata (suites not originally produced by `to_openeval()`) still convert,
    just with `financebench_id` set from the EvalPort `TestCase.id` and every
    other FinanceBench-specific field omitted rather than fabricated.
    """
    rows: List[Dict[str, Any]] = []
    for tc in suite.get("test_cases", []) or []:
        metadata = tc.get("metadata") or {}
        row: Dict[str, Any] = {"financebench_id": tc.get("id")}

        row["question"] = tc.get("input", "")
        if "expected_output" in tc:
            row["answer"] = tc["expected_output"]

        for meta_key, value in metadata.items():
            if not meta_key.startswith("financebench."):
                continue
            field = meta_key[len("financebench.") :]
            if field in _DOC_INFO_ONLY_FIELDS:
                # Document-info fields aren't part of financebench_open_source.jsonl
                # rows themselves (they live in the separate document-info file
                # this format joins on doc_name) -- skip them here so the round
                # trip reproduces the right *file's* shape, not a merged one.
                continue
            row[field] = value

        rows.append(row)
    return rows


def result_to_openeval(
    result_rows: Iterable[Dict[str, Any]],
    suite_id: str = "financebench_open_source",
    run_id: str = "financebench_results",
    started_at: str = "1970-01-01T00:00:00Z",
) -> Dict[str, Any]:
    """Convert FinanceBench `results/*.jsonl` rows (the shape of e.g.
    `results/gpt-4_sharedStore.jsonl`: `financebench_id`, `model_name`,
    `eval_mode`, `temp`, `question`, `gold_answer`, `model_answer`, `label`)
    into an EvalPort ResultSet (dict).

    `label` is FinanceBench's own human-annotated correctness judgment
    (observed values: "Correct Answer", "Incorrect Answer", "Refusal" -- any
    other string is passed through as `passed=False` rather than raising, in
    case FinanceBench adds a new label value upstream). This function carries
    that real human judgment into EvalPort's `Result.grader_results` (grader
    type `"human"`) rather than re-scoring anything itself -- no LLM call,
    no fabricated confidence score.

    `started_at` defaults to the Unix epoch since FinanceBench's result files
    don't record when the run happened; pass a real ISO-8601 timestamp if you
    have one for your own run.
    """
    results: List[Dict[str, Any]] = []
    for row in result_rows:
        fb_id = row.get("financebench_id")
        test_case_id = str(fb_id) if fb_id is not None else f"financebench_row_{len(results)}"

        label = row.get("label")
        passed = label in _PASSING_LABELS
        score = 1.0 if passed else 0.0

        metadata: Dict[str, Any] = {}
        for key in ("model_name", "eval_mode", "temp", "gold_answer"):
            if key in row:
                metadata[f"financebench.{key}"] = row[key]
        if label is not None and label not in _KNOWN_LABELS:
            metadata["financebench.unrecognized_label"] = True

        result_entry: Dict[str, Any] = {
            "test_case_id": test_case_id,
            "actual_output": row.get("model_answer", ""),
            "grader_results": [
                {
                    "grader_id": _HUMAN_GRADER_ID,
                    "type": "human",
                    "score": score,
                    "passed": passed,
                    "reason": str(label) if label is not None else "",
                }
            ],
            "passed": passed,
            "metadata": metadata,
        }
        results.append(result_entry)

    return {
        "version": _SPEC_VERSION,
        "suite_id": suite_id,
        "run_id": run_id,
        "started_at": started_at,
        "runner": {"name": "financebench-openeval-adapter"},
        "results": results,
    }
