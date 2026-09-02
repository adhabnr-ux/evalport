# halumem-openeval-adapter

Convert [MemTensor/HaluMem](https://github.com/MemTensor/HaluMem)'s operation-level QA, extraction, and update evaluation records to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

HaluMem is a benchmark for hallucination in AI memory systems, with **operation-level** granularity: it doesn't just score a final answer, it separately grades memory *extraction* (did the system capture the right memory points, without hallucinating extras?) and memory *update* (did the system correctly revise a memory when new information arrived?), alongside the more familiar QA layer. Built following the design worked out with the HaluMem maintainer in [MemTensor/HaluMem#12](https://github.com/MemTensor/HaluMem/issues/12).

## Install

```bash
pip install "halumem-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/halumem-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (the same install path already used by this repo's other adapters).

HaluMem's `eval/` pipeline is not a pip-installable package either — it's a set of scripts (`eval/evaluation.py`, `eval/eval_tools.py`, etc.) you run against a local checkout, which write plain JSON to disk. This adapter's only runtime dependency is `evalport-sdk`; there's no separate `[halumem]` extra to install.

## The four HaluMem operations

HaluMem's real evaluation output (`eval/evaluation.py`'s `main()`) is a single JSON object — not JSONL — shaped like:

```json
{
  "overall_score": {...},
  "memory_integrity_records": [...],
  "memory_accuracy_records": [...],
  "memory_update_records": [...],
  "question_answering_records": [...]
}
```

Every function in this module takes an `operation` argument — one of `"qa"`, `"memory_integrity"`, `"memory_accuracy"`, `"memory_update"` — and accepts either that whole dict (pulling the right record list out by key) or an already-extracted `list[dict]` directly.

| `operation` | Real HaluMem record list | What it measures |
|---|---|---|
| `"qa"` | `question_answering_records` | Did the memory system answer correctly, given its own extracted memories? (`result_type`: Correct / Hallucination / Omission) |
| `"memory_integrity"` | `memory_integrity_records` | Did extraction *recall* each expected memory point? (`memory_integrity_score`: 0/1/2) |
| `"memory_accuracy"` | `memory_accuracy_records` | Was each *extracted* memory actually supported (not hallucinated)? (`memory_accuracy_score`: 0/1/2, `is_included_in_golden_memories`) |
| `"memory_update"` | `memory_update_records` | Did a memory update correctly reflect new information? (`memory_update_type`: Correct / Hallucination / Omission / **Other**) |

```python
from halumem_openeval_adapter import load_eval_results, to_openeval, result_to_openeval
from openeval.validate import validate_suite, validate_result_set

eval_results = load_eval_results("results/zep-default/zep_eval_stat_result.json")

suite = to_openeval(eval_results, "qa", judge_model="gpt-4o-2024-08-06")
assert validate_suite(suite).valid

result_set = result_to_openeval(eval_results, "qa", suite_id=suite["id"], run_id="zep_run_1")
assert validate_result_set(result_set).valid
```

Run the same two calls with `operation="memory_integrity"`, `"memory_accuracy"`, or `"memory_update"` to convert the other three layers.

## Scope: what this adapter is (and is not)

**This is an interoperability / result-conversion layer for evaluation records HaluMem's own pipeline has already produced — it is not an alternative implementation of the official HaluMem evaluator.**

The `llm_judge` grader definitions this module attaches to a converted suite (`_QA_PROMPT`, `_MEMORY_INTEGRITY_PROMPT`, etc.) are condensed paraphrases of HaluMem's real rubrics (`EVALUATION_PROMPT_FOR_*` in `eval/eval_tools.py`), rewritten only to fit EvalPort's `{input}`/`{expected}`/`{output}` placeholder convention — they exist so a converted suite is *self-describing* about how its verdicts were originally produced, not so that re-running them reproduces HaluMem's official scoring. Concretely, re-running these grader definitions is **not guaranteed to reproduce the official HaluMem evaluation pipeline**, for two independent reasons:

- The prompts are condensed, not byte-identical to HaluMem's real rubric text.
- For `memory_integrity` and `memory_accuracy`, the full extracted-memory pool and dialogue actually graded against is not persisted in `eval_results` (see the limitation noted below) — so even a byte-identical prompt would have less context available than HaluMem's own evaluator had.

`to_openeval()` + `result_to_openeval()` on **already-scored** `eval_results` carries HaluMem's own finished verdict through untouched (`result_type` / `memory_update_type` / the `*_score` fields) — that path does not call an LLM and involves no re-grading at all. The grader definitions only become directly relevant if you choose to re-run evaluation using EvalPort tooling against the converted suite, which is a different (and lossier, per the two points above) operation from converting existing results, and isn't necessary for the common case of just moving already-computed HaluMem verdicts into a portable format.

**On `judge_model` when converting *existing* results specifically:** `result_to_openeval()`'s `judge_model` falls back to the current `OPENAI_MODEL` environment variable (see point 1 below) when no explicit value is given. That default is meant for the case where you're actively running HaluMem's evaluator and converting its output in the same environment/session, so `OPENAI_MODEL` genuinely reflects the model in use. If you're instead converting `eval_results` produced at an earlier time (a different session, a colleague's run, an artifact from before you changed `OPENAI_MODEL`), **pass `judge_model=` explicitly with whatever model actually produced those verdicts** — the environment variable at conversion time has no guaranteed relationship to the model that generated historical results, and silently trusting it would mislabel `provider.model` on the resulting `ResultSet`.

## How the four points raised in MemTensor/HaluMem#12 are addressed

1. **`type: "llm_judge"`, never `"human"`.** HaluMem's verdicts come from its own LLM-based evaluator (`eval/eval_tools.py`, via `eval/llms.py`), not a human grader. Every `GraderResult` this module emits uses `type: "llm_judge"`. The judge model is read from the real `OPENAI_MODEL` environment variable HaluMem's own `eval/llms.py` uses, unless a `judge_model=` argument overrides it — there is **no hardcoded default model name** anywhere in this module (unlike the original issue-proposal sketch, which defaulted to `"gpt-4o"`). Omitting both raises `ValueError` rather than guessing.

   ```python
   import os
   os.environ["OPENAI_MODEL"] = "gpt-4o-2024-08-06"
   suite = to_openeval(eval_results, "qa")  # reads OPENAI_MODEL
   # or:
   suite = to_openeval(eval_results, "qa", judge_model="gpt-4o-2024-08-06")  # explicit
   ```

2. **Categorical outcomes are never collapsed.** `Hallucination`/`Omission` (QA) and `Hallucination`/`Omission`/**`Other`** (update) all score `0.0`/`passed=False` — HaluMem only rewards the `"Correct"` outcome — but each `GraderResult.reason` is the literal HaluMem verdict string, duplicated onto a canonical `metadata.halumem.result_type` / `metadata.halumem.memory_update_type` key. A consumer never has to parse `reason` text to recover which of the three (or four) outcomes actually happened.

3. **Extraction scoring semantics preserved verbatim.** `memory_integrity_score`, `memory_accuracy_score`, `is_included_in_golden_memories` (kept as HaluMem's own literal `"true"`/`"false"` string, not silently coerced), `importance`, and `memory_source` (which is how HaluMem marks an "interference"/distractor memory) all land untouched in `metadata.halumem.*`, plus the normalized `[0,1]` score in `metadata.openeval.raw_score` per the EvalPort spec's own reserved-metadata convention for non-`[0,1]`-native scales. `tests/test_adapter.py::test_recomputed_aggregates_match_real_halumem_formula` recomputes HaluMem's own `memory_integrity`/`memory_accuracy`/`memory_extraction_f1`/`memory_update` aggregate ratios **purely from the converted `ResultSet`s** and diffs them against a real `eval_results["overall_score"]` shape — the maintainer's own stated acceptance criterion.

4. **Stable, digest-based IDs.** Every test-case/result ID is a SHA-256 digest over HaluMem's own `operation` + `uuid` + `ssession_id` + the record's position within its record list + the record's real content field (question or memory content) — see `_stable_id()`. Same input, same ID, every time, in a fresh process — unlike Python's built-in `hash()`, which is per-process-randomized for strings. `tests/test_adapter.py::test_stable_id_matches_across_fresh_interpreter_process` proves this by literally spawning a second interpreter with hash randomization left on its default and comparing IDs.

## Follow-up fixes from the maintainer's second review (2026-08-31)

After the adapter above was merged, the maintainer flagged two more things, both addressed:

- **Scope framing and judge-model guidance.** See "Scope: what this adapter is (and is not)" above — this adapter is documented as an interoperability/result-conversion layer, not a reimplementation of HaluMem's official evaluator, and the README now explicitly recommends passing `judge_model=` explicitly (rather than relying on `OPENAI_MODEL`) when converting `eval_results` produced at an earlier time or by someone else.
- **ID collision on duplicate content, fixed.** The digest in point 4 above originally hashed only `operation + uuid + ssession_id + content`, so two records with identical content in the same user/session/operation (e.g. two distinct `memory_accuracy` candidates that happen to extract the same phrase from two different turns) collided onto the same ID. `_stable_id()` now also folds in the record's position within the operation's record list, disambiguating that case while remaining a deterministic function of the input — not an incrementing process-local counter. `tests/test_adapter.py::test_duplicate_content_in_same_session_gets_distinct_ids` constructs exactly that collision case (same `uuid`/`ssession_id`/`memory_content`, different scores) and confirms both the `to_openeval()` test-case IDs and the corresponding `result_to_openeval()` result IDs come out distinct and correctly paired; `test_duplicate_content_ids_still_stable_and_digest_based` confirms the fix didn't turn IDs into something order-of-calls-dependent rather than input-dependent.

## A real limitation of the extraction operations, stated plainly

For `"memory_integrity"` and `"memory_accuracy"`, HaluMem's real records (confirmed by reading `eval/evaluation.py`'s `process_user()`) do **not** persist the full extracted-memory pool or dialogue text that was actually graded against — only the golden/candidate memory point and its score survive into `eval_results`. This adapter does not fabricate an `actual_output` or `context` for those two operations; `Result.actual_output` is left unset rather than invented. `"memory_update"` doesn't have this gap — HaluMem does persist `memories_from_system`, so that operation's `Result.actual_output` is real.

## Interference memories: an inverted pass condition

A `memory_source == "interference"` record is a distractor the system should **not** have recalled, so `passed` for those records means `memory_integrity_score == 0` (correctly resisted), the opposite of the `== 2` ("fully recalled") condition for real golden memory points — mirroring the asymmetric `interference_memory_scores` counter in `eval/evaluation.py`'s own `aggregate_eval_results()`. See `tests/test_adapter.py::test_memory_integrity_interference_pass_condition_is_inverted`.

## Grader prompts

HaluMem's real evaluation prompts (`EVALUATION_PROMPT_FOR_*` in `eval/eval_tools.py`) use placeholders like `{memories}` / `{expected_memory_point}` / `{question}` that don't match EvalPort's `{input}`/`{expected}`/`{output}` convention (required by `spec/SPEC.md` Validation Rule 4 for `llm_judge` graders). Each grader's `params.prompt` in this module is a condensed paraphrase of HaluMem's real rubric text, rewritten only enough to use EvalPort's placeholder convention — not independently invented criteria — and is labeled in a source comment with which real prompt constant it's adapted from.

## `from_openeval()` — round trip

```python
from halumem_openeval_adapter import from_openeval

rebuilt = from_openeval(suite, "memory_update")  # -> list[dict], HaluMem record shape
```

Round-trips every field this adapter itself writes into `TestCase.metadata` (prefixed `halumem.*`) or `TestCase.context` back to its original HaluMem field name. Suites not originally produced by this adapter still convert — you just get the identifying fields back, with no fabricated values for fields that were never there.

## Licensing note

The HaluMem repository and dataset are released under **CC BY-NC-ND 4.0**. This adapter operates on a user's own locally-generated HaluMem `eval_results` output and does **not** redistribute any HaluMem code or dataset content — `tests/fixtures/synthetic_eval_results.json` is entirely hand-written to match the real record shapes documented in `eval/evaluation.py` / `eval/eval_tools.py`, not downloaded or derived from real HaluMem dataset rows.

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -v
```

52 tests, all run against a real `evalport-sdk` installation (`openeval.validate.validate_suite()` / `validate_result_set()`, not a mock) — covering all four operations' `to_openeval()`/`result_to_openeval()`/`from_openeval()`, the four MemTensor/HaluMem#12 feedback points above, the two follow-up fixes below, and the aggregate-metric round-trip.

## License

Apache 2.0, matching the rest of this repo. (HaluMem itself, referenced but not redistributed, is CC BY-NC-ND 4.0 — see Licensing note above.)
