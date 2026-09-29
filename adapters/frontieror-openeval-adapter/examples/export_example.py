"""Rebuild the example export in examples/output/ from the real FrontierOR
run CSVs in examples/data/ (see the adapter README, "Example export").

    python examples/export_example.py                       # built-in scorer
    python examples/export_example.py --frontieror-root ../FrontierOR   # FrontierOR's own StagedQteScorer
    python examples/export_example.py --data-dir ../FrontierOR/frontier-or --suite   # also rebuild suite.json

Only public data is used: the reference rows in data/gurobi_references_subset.csv
are copied verbatim from the Hugging Face dataset's public
metadata/gurobi_references.csv.gz.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import frontieror_openeval_adapter as A  # noqa: E402

DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
RUN_CSVS = ["eval_results_quickstart.csv", "eval_results_quickstart_gurobi_tiny.csv"]
PAPERS = ["bierwirth2017", "liao2020", "walteros2020"]
# From data/RUN.json: start of the first one_shot_eval.py invocation and
# the time the second one wrote its CSV.
STARTED_AT = "2026-09-29T18:24:54Z"
COMPLETED_AT = "2026-09-29T18:34:03Z"
GROUP_ID = "frontieror-quickstart-samples-2026-09-29"
DATASET_REVISION = "SmartOR/FrontierOR@9f5cd5436f7f4e013dfa81aebea77a1cc98628b8"
# one_shot_eval.py's --time_limit default (Quick Start passes none), used for
# every instance in both invocations.
CANDIDATE_TIME_LIMIT = 300
# Rows whose runtime_error came from this environment (pip gurobipy's
# size-limited license, "Model too large for size-limited license" in the run
# log), not from the program. Kept verbatim in the CSV, skipped here.
ENVIRONMENT_FAILURES = {
    ("liao2020", "claude-opus-4.6", "tiny"),
    ("walteros2020", "llama-4-maverick", "tiny"),
    ("walteros2020", "qwen3-coder-plus", "tiny"),
}


def build_result_sets(staged_qte_source="auto", frontieror_root=None):
    refs = A.load_references(os.path.join(DATA, "gurobi_references_subset.csv"))
    meta = A.load_paper_meta(os.path.join(DATA, "paper_meta_subset.json"))
    rows = []
    for name in RUN_CSVS:
        rows.extend(
            r for r in A.read_results_csv(os.path.join(DATA, name))
            if (r["paper_id"], r["model"], r["instance"]) not in ENVIRONMENT_FAILURES
        )
    out = {}
    models = A.split_by_model(rows)
    for seq, model in enumerate(sorted(models)):
        out[f"resultset_{model}.json"] = A.results_to_openeval(
            models[model],
            references=refs,
            paper_meta=meta,
            suite_id="frontieror_quickstart_samples",
            run_id=f"{GROUP_ID}/{model}",
            started_at=STARTED_AT,
            completed_at=COMPLETED_AT,
            candidate_time_limits=CANDIDATE_TIME_LIMIT,
            staged_qte_source=staged_qte_source,
            frontieror_root=frontieror_root,
            runner={"name": "FrontierOR one_shot_eval.py (--reuse-code all --code-root samples/oneshot_code --exec-mode bare)", "version": "Minw913/FrontierOR@bc1bdde"},
            group={"group_id": GROUP_ID, "role": "model", "label": model, "sequence": seq},
            dataset_revision=DATASET_REVISION,
        )
    return out


def build_suite(data_dir):
    meta = A.load_paper_meta(os.path.join(DATA, "paper_meta_subset.json"))
    items = A.load_dataset_instances(data_dir, PAPERS, instances=("tiny", "large_1"))
    return A.to_openeval(
        items,
        paper_meta=meta,
        suite_id="frontieror_quickstart_samples",
        name="FrontierOR quick-start sample papers",
        description="Public FrontierOR tasks bierwirth2017, liao2020, walteros2020 (tiny, large_1: the instances the example run covered). CC-BY-4.0, SmartOR/FrontierOR.",
        dataset_revision=DATASET_REVISION,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontieror-root", default=None)
    parser.add_argument("--data-dir", default=None, help="local SmartOR/FrontierOR download, for --suite")
    parser.add_argument("--suite", action="store_true")
    args = parser.parse_args()
    os.makedirs(OUT, exist_ok=True)
    source = "upstream" if args.frontieror_root else "builtin"
    for fname, rs in build_result_sets(source, args.frontieror_root).items():
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as fh:
            json.dump(rs, fh, indent=2)
            fh.write("\n")
        print("wrote", fname)
    if args.suite:
        if not args.data_dir:
            parser.error("--suite needs --data-dir")
        with open(os.path.join(OUT, "suite.json"), "w", encoding="utf-8") as fh:
            json.dump(build_suite(args.data_dir), fh, indent=2)
            fh.write("\n")
        print("wrote suite.json")


if __name__ == "__main__":
    main()
