# EvalPort Benchmark Hub

14 well-known public benchmarks, converted into 22 individually-valid
[EvalPort](https://github.com/adhabnr-ux/evalport) suites — 8,012 test
cases total, every one of them passing `openeval.validate.validate_suite()`.
This is the reference set of "real" suites for anyone building or testing
an EvalPort-compatible runner, adapter, or tool: pull any file below and
you have a spec-conformant suite backed by a real, cited, license-cleared
benchmark, not a synthetic example.

## Quickstart

```bash
# Validate a single suite against the real EvalPort validator
pip install evalport-sdk
python3 -c "
import json
from openeval.validate import validate_suite
suite = json.load(open('benchmarks/gsm8k/gsm8k.json'))
print(validate_suite(suite).valid)
"

# Or validate every suite in the hub at once (the same check CI runs)
python3 benchmarks/_tools/validate_all.py
```

Once `evalport run` (the CLI runner, see the main repo README) lands, this
whole hub becomes runnable directly:

```bash
evalport run benchmarks/gsm8k/gsm8k.json --provider openai --model gpt-4o-mini --dry-run
```

## Index

| Benchmark | Suite file(s) | Cases | License | Type |
|---|---|---|---|---|
| [GSM8K](gsm8k/) | `gsm8k/gsm8k.json` | 500 | MIT | Math word problems |
| [ARC-Easy](arc-easy/) | `arc-easy/arc-easy.json` | 500 | CC-BY-SA-4.0 | Science MC |
| [ARC-Challenge](arc-challenge/) | `arc-challenge/arc-challenge.json` | 500 | CC-BY-SA-4.0 | Science MC (harder) |
| [BoolQ](boolq/) | `boolq/boolq.json` | 500 | CC-BY-SA-3.0 | Yes/no reading comprehension |
| [HellaSwag](hellaswag/) | `hellaswag/hellaswag.json` | 500 | MIT | Commonsense sentence completion |
| [WinoGrande](winogrande/) | `winogrande/winogrande.json` | 500 | CC-BY | Pronoun resolution |
| [CommonsenseQA](commonsenseqa/) | `commonsenseqa/commonsenseqa.json` | 500 | MIT | Commonsense MC |
| [PIQA](piqa/) | `piqa/piqa.json` | 500 | AFL-3.0 | Physical commonsense |
| [TruthfulQA](truthfulqa/) | `truthfulqa/truthfulqa.json` | 817 | Apache-2.0 | Truthfulness (open-ended) |
| [MMLU](mmlu/) | 6 files, one per subject | 1,044 | MIT | Academic knowledge MC |
| [HumanEval](humaneval/) | `humaneval/humaneval.json` | 164 | MIT | Python code generation |
| [MBPP](mbpp/) | `mbpp/mbpp.json` | 300 | CC-BY-4.0 | Python code generation |
| [SQuAD 2.0](squad2/) | `squad2/squad2.json` | 500 | CC-BY-SA-4.0 | Reading comprehension (+ unanswerable) |
| [DROP](drop/) | `drop/drop.json` | 500 | CC-BY-SA-4.0 | Discrete reasoning over paragraphs |
| [BBH](bbh/) | 3 files, one per task | 687 | MIT | Hard multi-step reasoning |
| **Total** | **22 suite files** | **8,012** | | |

Every benchmark's license was verified from its primary source (dataset
card or repo `LICENSE`) before inclusion — see [LICENSES.md](LICENSES.md)
for the full attribution table, licensing notes, and one candidate
(LAMBADA) that was deliberately excluded over unclear redistribution
rights.

## How suites are structured

Every suite is a single JSON file conforming to the EvalPort `EvalSuite`
schema (`version`, `id`, `name`, `test_cases`, `graders`, `metadata`). Two
grader architectures are used, depending on whether a benchmark's grading
logic is constant across all cases or varies per case:

- **Shared suite-level graders** (GSM8K, ARC, BoolQ, HellaSwag, WinoGrande,
  CommonsenseQA, PIQA, TruthfulQA, MMLU, BBH) — a single grader definition
  referenced by id from every test case, used when every case in the suite
  is graded the same way (e.g. `exact_match` on a letter/number/boolean, or
  a fixed `semantic_similarity` threshold).
- **Per-case inline graders** (HumanEval, MBPP, SQuAD 2.0, DROP) — each test
  case embeds its own grader dict directly, used when grading parameters
  genuinely vary per case (a unique unit-test harness per coding problem, a
  unique expected substring per QA pair). This is a deliberate architecture
  choice driven by reading the SDK's own validator: `validate_suite`'s
  `DANGLING_REFERENCE` check only fires for string-type grader references
  in test cases, so a shared suite-level grader has no way to hold
  per-case-varying parameters — inline dict graders are the only spec-valid
  way to express this.

Every suite also carries a consistent metadata block for provenance and
reproducibility:

```json
"metadata": {
  "openeval": {"source": "evalport-benchmarks"},
  "evalport.source": "<link to the dataset's own source>",
  "evalport.source_license": "<verified license>",
  "evalport.original_paper": "<arXiv link>",
  "evalport.converted_by": "evalport-benchmarks/_tools/convert_hf_dataset.py",
  "evalport.conversion_date": "<date this suite was generated>",
  "evalport.case_count": <int, derived from the actual written cases>
}
```

## The conversion pipeline

All suites in this hub — except `_tools/convert_jsonl.py`-based ones, none
of which exist yet — are produced by one reusable script,
[`_tools/convert_hf_dataset.py`](_tools/convert_hf_dataset.py), rather than
20 one-off conversion scripts. It pulls each benchmark from HuggingFace
Datasets, applies benchmark-specific field mapping (documented per-handler
in the script itself and per-benchmark in each subdirectory's README),
**validates every suite against the real `openeval.validate.validate_suite()`
before writing it to disk**, and refuses to write anything that fails
validation.

```bash
cd _tools
python3 convert_hf_dataset.py gsm8k              # convert one benchmark (uses its default --limit)
python3 convert_hf_dataset.py mmlu --limit 50     # override the case limit
python3 convert_hf_dataset.py all                 # regenerate the entire hub
python3 validate_all.py                           # validate every suite under benchmarks/ (the CI gate)
```

Adding a new benchmark means adding one handler function + one `REGISTRY`
entry to `convert_hf_dataset.py`, not a new script.

## Per-benchmark case limits

Most suites cap at 500 cases (a size chosen to keep suite files small and
git-diff-friendly while still being a meaningful sample) even when the
source split is larger; TruthfulQA (817) and HumanEval (164) are included
in full since their native splits are already smaller than the default cap.
Every README linked in the index above states the exact case count and, for
capped suites, how many cases were available upstream.

## Contributing a new benchmark

1. Verify the benchmark's license permits data redistribution — check the
   HuggingFace dataset card's "Licensing Information" section or the
   source repo's `LICENSE` file directly. Don't assume from a benchmark's
   popularity or how another framework classifies it.
2. Add a handler function to `_tools/convert_hf_dataset.py` (or
   `_tools/convert_jsonl.py` if the source isn't HF-hosted) plus a
   `REGISTRY` entry.
3. Run it, confirm `valid=True` in the script's own output.
4. Add an entry to this table, a per-benchmark `README.md` in the new
   subdirectory (source/license/citation/grader rationale/case count —
   match the shape of any existing benchmark README), and an entry to
   [LICENSES.md](LICENSES.md).
5. Run `_tools/validate_all.py` to confirm the whole hub — not just your
   new suite — still passes.
