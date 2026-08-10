#!/usr/bin/env python3
"""Convert well-known HuggingFace-hosted benchmarks into EvalPort suites.

One reusable script, many benchmark handlers, so adding the next benchmark
means adding one function here rather than a new one-off script. Every
handler returns a plain EvalPort EvalSuite dict; `main()` validates it
against the real SDK validator before writing it to disk, and refuses to
write a suite that doesn't pass.

Usage:
    python3 convert_hf_dataset.py <benchmark_key> [--limit N]
    python3 convert_hf_dataset.py all   # convert every registered benchmark

Requires: `datasets` (pip install datasets), and evalport-sdk importable
(the sdk/python directory of this repo, or an installed evalport-sdk).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk" / "python"))

from openeval.validate import validate_suite  # noqa: E402
from openeval.types import OPENEVAL_VERSION  # noqa: E402

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent

LETTERS = "ABCDEFGHIJ"

# Metadata discipline: every suite this pipeline produces must carry these
# two fields in addition to whatever per-benchmark source_meta a handler
# supplies (evalport.source, evalport.source_license, evalport.original_paper,
# evalport.case_count). CONVERSION_DATE is fixed per pipeline run rather than
# computed from wall-clock time so re-running the script for a fix doesn't
# spuriously touch the date on unrelated suites re-generated the same day;
# bump it by hand if you run a fresh conversion pass on a different day.
CONVERSION_DATE = "2026-08-10"
CONVERTED_BY = "evalport-benchmarks/_tools/convert_hf_dataset.py"


def _exact_match_grader() -> Dict[str, Any]:
    return {"id": "gr_exact_match", "type": "exact_match", "params": {"ignore_case": True, "strip": True}}


def _semantic_similarity_grader(threshold: float = 0.8) -> Dict[str, Any]:
    return {
        "id": "gr_semantic_similarity",
        "type": "semantic_similarity",
        "params": {"threshold": threshold, "model": "text-embedding-3-small"},
    }


def _mc_test_case(tc_id: str, question: str, choices: List[str], correct_index: int, **extra) -> Dict[str, Any]:
    """Build a multiple-choice test case: question + lettered choices, expected_output = correct letter."""
    lettered = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    tc = {
        "id": tc_id,
        "input": f"{question}\n\n{lettered}",
        "expected_output": LETTERS[correct_index],
        "graders": ["gr_exact_match"],
    }
    tc.update(extra)
    return tc


def _suite(benchmark_id: str, name: str, test_cases: List[Dict[str, Any]], graders: List[Dict[str, Any]], source_meta: Dict[str, Any]) -> Dict[str, Any]:
    full_meta = {
        "evalport.converted_by": CONVERTED_BY,
        "evalport.conversion_date": CONVERSION_DATE,
        **source_meta,
    }
    # case_count is derived from the actual written test_cases, not a value
    # a handler might pass in and forget to update if --limit changes it.
    full_meta["evalport.case_count"] = len(test_cases)
    return {
        "version": OPENEVAL_VERSION,
        "id": benchmark_id,
        "name": name,
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "evalport-benchmarks"}, **full_meta},
    }


# ---------------------------------------------------------------------------
# Benchmark handlers. Each returns (relative_output_path, suite_dict).
# ---------------------------------------------------------------------------

def convert_gsm8k(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        # The reference solution ends with "#### <final numeric answer>".
        m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", row["answer"])
        final_answer = m.group(1).replace(",", "") if m else row["answer"].strip()
        test_cases.append({
            "id": f"gsm8k_{i}",
            "input": row["question"],
            "expected_output": final_answer,
            "graders": ["gr_exact_match"],
            "metadata": {"gsm8k_full_solution": row["answer"]},
        })
    suite = _suite(
        "bench_gsm8k",
        "GSM8K — Grade School Math",
        test_cases,
        [_exact_match_grader()],
        {
            "evalport.source": "https://github.com/openai/grade-school-math",
            "evalport.source_license": "MIT",
            "evalport.original_paper": "https://arxiv.org/abs/2110.14168",
            "evalport.case_count": len(test_cases),
        },
    )
    return [("gsm8k/gsm8k.json", suite)]


def convert_arc(limit: int) -> List[tuple]:
    from datasets import load_dataset

    out = []
    for config, suffix in (("ARC-Easy", "easy"), ("ARC-Challenge", "challenge")):
        ds = load_dataset("allenai/ai2_arc", config, split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        test_cases = []
        skipped = 0
        for i, row in enumerate(ds):
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            answer_key = row["answerKey"]
            if answer_key not in labels or len(texts) < 2:
                skipped += 1
                continue
            correct_index = labels.index(answer_key)
            test_cases.append(_mc_test_case(f"arc_{suffix}_{i}", row["question"], texts, correct_index))
        suite = _suite(
            f"bench_arc_{suffix}",
            f"AI2 Reasoning Challenge — {config}",
            test_cases,
            [_exact_match_grader()],
            {
                "evalport.source": "https://huggingface.co/datasets/allenai/ai2_arc",
                "evalport.source_license": "CC-BY-SA-4.0",
                "evalport.original_paper": "https://arxiv.org/abs/1803.05457",
                "evalport.case_count": len(test_cases),
                "evalport.skipped_malformed_cases": skipped,
            },
        )
        out.append((f"arc-{suffix}/arc-{suffix}.json", suite))
    return out


def convert_boolq(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        test_cases.append({
            "id": f"boolq_{i}",
            "input": f"{row['passage']}\n\nQuestion: {row['question']}?\nAnswer with exactly \"true\" or \"false\".",
            "expected_output": "true" if row["answer"] else "false",
            "graders": ["gr_exact_match"],
        })
    suite = _suite(
        "bench_boolq",
        "BoolQ — Yes/No Reading Comprehension",
        test_cases,
        [_exact_match_grader()],
        {
            "evalport.source": "https://github.com/google-research-datasets/boolean-questions",
            "evalport.source_license": "CC-BY-SA-3.0",
            "evalport.original_paper": "https://arxiv.org/abs/1905.10044",
            "evalport.case_count": len(test_cases),
        },
    )
    return [("boolq/boolq.json", suite)]


def convert_hellaswag(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        label = row["label"]
        if label == "" or label is None:
            continue
        correct_index = int(label)
        context = f"{row['activity_label']}: {row['ctx']}"
        test_cases.append(_mc_test_case(f"hellaswag_{i}", f"Which ending best completes this scenario?\n{context}", row["endings"], correct_index))
    suite = _suite(
        "bench_hellaswag",
        "HellaSwag — Commonsense Sentence Completion",
        test_cases,
        [_exact_match_grader()],
        {
            "evalport.source": "https://github.com/rowanz/hellaswag",
            "evalport.source_license": "MIT",
            "evalport.original_paper": "https://arxiv.org/abs/1905.07830",
            "evalport.case_count": len(test_cases),
        },
    )
    return [("hellaswag/hellaswag.json", suite)]


def convert_winogrande(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        answer = row["answer"]
        if answer not in ("1", "2"):
            continue
        correct_index = int(answer) - 1
        choices = [row["option1"], row["option2"]]
        prompt = f"Fill in the blank ('_') in this sentence:\n{row['sentence']}"
        test_cases.append(_mc_test_case(f"winogrande_{i}", prompt, choices, correct_index))
    suite = _suite(
        "bench_winogrande",
        "WinoGrande — Pronoun Resolution",
        test_cases,
        [_exact_match_grader()],
        {
            "evalport.source": "https://github.com/allenai/winogrande",
            "evalport.source_license": "CC-BY",
            "evalport.original_paper": "https://arxiv.org/abs/1907.10641",
            "evalport.case_count": len(test_cases),
        },
    )
    return [("winogrande/winogrande.json", suite)]


def convert_commonsenseqa(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("tau/commonsense_qa", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        answer_key = row["answerKey"]
        if answer_key not in labels:
            continue
        correct_index = labels.index(answer_key)
        test_cases.append(_mc_test_case(f"commonsenseqa_{i}", row["question"], texts, correct_index))
    suite = _suite(
        "bench_commonsenseqa",
        "CommonsenseQA",
        test_cases,
        [_exact_match_grader()],
        {
            "evalport.source": "https://huggingface.co/datasets/tau/commonsense_qa",
            "evalport.source_license": "MIT",
            "evalport.original_paper": "https://arxiv.org/abs/1811.00937",
            "evalport.case_count": len(test_cases),
        },
    )
    return [("commonsenseqa/commonsenseqa.json", suite)]


def convert_piqa(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        label = row["label"]
        if label not in (0, 1):
            continue
        choices = [row["sol1"], row["sol2"]]
        test_cases.append(_mc_test_case(f"piqa_{i}", f"Goal: {row['goal']}\nWhich solution is more physically sensible?", choices, label))
    suite = _suite(
        "bench_piqa",
        "PIQA — Physical Interaction QA",
        test_cases,
        [_exact_match_grader()],
        {
            "evalport.source": "https://yonatanbisk.com/piqa/",
            "evalport.source_license": "AFL-3.0",
            "evalport.original_paper": "https://arxiv.org/abs/1911.11641",
            "evalport.case_count": len(test_cases),
        },
    )
    return [("piqa/piqa.json", suite)]


def convert_truthfulqa(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        best = row.get("best_answer") or (row["correct_answers"][0] if row["correct_answers"] else None)
        if not best:
            continue
        test_cases.append({
            "id": f"truthfulqa_{i}",
            "input": row["question"],
            "expected_output": best,
            "graders": ["gr_semantic_similarity"],
            "metadata": {"truthfulqa_category": row.get("category"), "truthfulqa_all_correct": row["correct_answers"]},
        })
    suite = _suite(
        "bench_truthfulqa",
        "TruthfulQA — Truthfulness",
        test_cases,
        [_semantic_similarity_grader(0.8)],
        {
            "evalport.source": "https://huggingface.co/datasets/truthful_qa",
            "evalport.source_license": "Apache-2.0",
            "evalport.original_paper": "https://arxiv.org/abs/2109.07958",
            "evalport.case_count": len(test_cases),
            "evalport.grading_note": "semantic_similarity is a coarse proxy for truthfulness; the TruthfulQA authors' own metric uses fine-tuned judges (GPT-judge/GPT-info) which this suite does not attempt to reproduce.",
        },
    )
    return [("truthfulqa/truthfulqa.json", suite)]


MMLU_SUBJECTS = [
    "high_school_mathematics",
    "high_school_us_history",
    "high_school_computer_science",
    "professional_law",
    "college_biology",
    "moral_scenarios",
]


def convert_mmlu(limit: int) -> List[tuple]:
    from datasets import load_dataset

    out = []
    for subject in MMLU_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject, split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        test_cases = []
        for i, row in enumerate(ds):
            test_cases.append(_mc_test_case(f"mmlu_{subject}_{i}", row["question"], row["choices"], row["answer"]))
        suite = _suite(
            f"bench_mmlu_{subject}",
            f"MMLU — {subject.replace('_', ' ').title()}",
            test_cases,
            [_exact_match_grader()],
            {
                "evalport.source": "https://github.com/hendrycks/test",
                "evalport.source_license": "MIT",
                "evalport.original_paper": "https://arxiv.org/abs/2009.03300",
                "evalport.case_count": len(test_cases),
                "evalport.mmlu_subject": subject,
            },
        )
        out.append((f"mmlu/mmlu-{subject.replace('_', '-')}.json", suite))
    return out


def convert_humaneval(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for row in ds:
        source = row["prompt"] + "\n{{completion}}\n" + row["test"] + f"\ncheck({row['entry_point']})\n"
        test_cases.append({
            "id": row["task_id"].replace("/", "_"),
            "input": row["prompt"],
            "graders": [{
                "id": f"gr_code_{row['task_id'].replace('/', '_')}",
                "type": "code",
                "description": "Runs the HumanEval canonical unit test harness against the model's completion.",
                "params": {"language": "python", "source": source, "entry_point": row["entry_point"]},
            }],
            "metadata": {"humaneval_canonical_solution": row["canonical_solution"]},
        })
    suite = _suite(
        "bench_humaneval",
        "HumanEval — Python Code Generation",
        test_cases,
        [],
        {
            "evalport.source": "https://github.com/openai/human-eval",
            "evalport.source_license": "MIT",
            "evalport.original_paper": "https://arxiv.org/abs/2107.03374",
            "evalport.case_count": len(test_cases),
            "evalport.grading_note": "The `code` grader requires a sandboxed Python execution environment; runners without one should skip these test cases per the EvalPort spec's unsupported_grader_type mechanism rather than eval() untrusted output directly.",
        },
    )
    return [("humaneval/humaneval.json", suite)]


def convert_mbpp(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for row in ds:
        tests_joined = "\n".join(row["test_list"])
        source = "{{completion}}\n" + tests_joined
        test_cases.append({
            "id": f"mbpp_{row['task_id']}",
            "input": f"{row['text']}\n\nYour solution must satisfy:\n" + "\n".join(row["test_list"]),
            "graders": [{
                "id": f"gr_code_mbpp_{row['task_id']}",
                "type": "code",
                "description": "Runs MBPP's asserted test list against the model's completion.",
                "params": {"language": "python", "source": source},
            }],
            "metadata": {"mbpp_reference_code": row["code"]},
        })
    suite = _suite(
        "bench_mbpp",
        "MBPP — Mostly Basic Python Problems",
        test_cases,
        [],
        {
            "evalport.source": "https://github.com/google-research/google-research/tree/master/mbpp",
            "evalport.source_license": "CC-BY-4.0",
            "evalport.original_paper": "https://arxiv.org/abs/2108.07732",
            "evalport.case_count": len(test_cases),
            "evalport.grading_note": "The `code` grader requires a sandboxed Python execution environment; runners without one should skip these test cases per the EvalPort spec's unsupported_grader_type mechanism.",
        },
    )
    return [("mbpp/mbpp.json", suite)]


def convert_squad2(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        answers = row["answers"]["text"]
        expected = answers[0] if answers else "unanswerable"
        substring = expected if answers else "unanswerable"
        test_cases.append({
            "id": f"squad2_{i}",
            "input": f"{row['context']}\n\nQuestion: {row['question']}\nIf the question cannot be answered from the passage, respond \"unanswerable\".",
            "expected_output": expected,
            "graders": [{
                "id": f"gr_contains_squad2_{i}",
                "type": "contains",
                "params": {"substring": substring, "ignore_case": True},
            }],
        })
    suite = _suite(
        "bench_squad2",
        "SQuAD 2.0 — Reading Comprehension",
        test_cases,
        [],
        {
            "evalport.source": "https://rajpurkar.github.io/SQuAD-explorer/",
            "evalport.source_license": "CC-BY-SA-4.0",
            "evalport.original_paper": "https://arxiv.org/abs/1806.03822",
            "evalport.case_count": len(test_cases),
            "evalport.grading_note": "Uses a substring-containment check against one reference answer as a lightweight proxy for SQuAD's official token-level F1/EM metric, which this suite does not reproduce.",
        },
    )
    return [("squad2/squad2.json", suite)]


def convert_drop(limit: int) -> List[tuple]:
    from datasets import load_dataset

    ds = load_dataset("ucinlp/drop", split="validation")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    test_cases = []
    for i, row in enumerate(ds):
        spans = row["answers_spans"]["spans"]
        expected = spans[0] if spans else None
        if not expected:
            continue
        test_cases.append({
            "id": f"drop_{i}",
            "input": f"{row['passage']}\n\nQuestion: {row['question']}",
            "expected_output": expected,
            "graders": [{
                "id": f"gr_contains_drop_{i}",
                "type": "contains",
                "params": {"substring": expected, "ignore_case": True},
            }],
        })
    suite = _suite(
        "bench_drop",
        "DROP — Discrete Reasoning Over Paragraphs",
        test_cases,
        [],
        {
            "evalport.source": "https://huggingface.co/datasets/ucinlp/drop",
            "evalport.source_license": "CC-BY-SA-4.0",
            "evalport.original_paper": "https://arxiv.org/abs/1903.00161",
            "evalport.case_count": len(test_cases),
            "evalport.grading_note": "Uses a substring-containment check against one reference answer span as a lightweight proxy for DROP's official F1 metric, which this suite does not reproduce.",
        },
    )
    return [("drop/drop.json", suite)]


BBH_TASKS = ["logical_deduction_five_objects", "causal_judgement", "date_understanding"]


def convert_bbh(limit: int) -> List[tuple]:
    from datasets import load_dataset

    out = []
    for task in BBH_TASKS:
        ds = load_dataset("lukaemon/bbh", task, split="test")
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        test_cases = []
        for i, row in enumerate(ds):
            test_cases.append({
                "id": f"bbh_{task}_{i}",
                "input": row["input"],
                "expected_output": row["target"],
                "graders": ["gr_exact_match"],
            })
        suite = _suite(
            f"bench_bbh_{task}",
            f"BIG-Bench Hard — {task.replace('_', ' ').title()}",
            test_cases,
            [_exact_match_grader()],
            {
                "evalport.source": "https://github.com/suzgunmirac/BIG-Bench-Hard",
                "evalport.source_license": "MIT",
                "evalport.original_paper": "https://arxiv.org/abs/2210.09261",
                "evalport.case_count": len(test_cases),
                "evalport.bbh_task": task,
            },
        )
        out.append((f"bbh/bbh-{task.replace('_', '-')}.json", suite))
    return out


REGISTRY: Dict[str, Callable[[int], List[tuple]]] = {
    "gsm8k": convert_gsm8k,
    "arc": convert_arc,
    "boolq": convert_boolq,
    "hellaswag": convert_hellaswag,
    "winogrande": convert_winogrande,
    "commonsenseqa": convert_commonsenseqa,
    "piqa": convert_piqa,
    "truthfulqa": convert_truthfulqa,
    "mmlu": convert_mmlu,
    "humaneval": convert_humaneval,
    "mbpp": convert_mbpp,
    "squad2": convert_squad2,
    "drop": convert_drop,
    "bbh": convert_bbh,
}

DEFAULT_LIMITS = {
    "gsm8k": 500, "arc": 500, "boolq": 500, "hellaswag": 500, "winogrande": 500,
    "commonsenseqa": 500, "piqa": 500, "truthfulqa": 0, "mmlu": 200,
    "humaneval": 0, "mbpp": 300, "squad2": 500, "drop": 500, "bbh": 250,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <benchmark_key|all> [--limit N]")
        sys.exit(1)
    key = sys.argv[1]
    limit_override = None
    if "--limit" in sys.argv:
        limit_override = int(sys.argv[sys.argv.index("--limit") + 1])

    keys = list(REGISTRY.keys()) if key == "all" else [key]
    for k in keys:
        if k not in REGISTRY:
            print(f"Unknown benchmark: {k}")
            continue
        limit = limit_override if limit_override is not None else DEFAULT_LIMITS.get(k, 500)
        print(f"=== Converting {k} (limit={limit or 'none'}) ===")
        try:
            results = REGISTRY[k](limit)
        except Exception as e:
            print(f"FAILED to convert {k}: {e}")
            raise
        for rel_path, suite in results:
            validation = validate_suite(suite)
            if not validation.valid:
                print(f"VALIDATION FAILED for {rel_path}: {validation.errors}")
                raise SystemExit(1)
            out_path = BENCHMARKS_DIR / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(suite, f, indent=2, ensure_ascii=False)
            print(f"  wrote {rel_path}  ({len(suite['test_cases'])} cases, valid=True)")


if __name__ == "__main__":
    main()
