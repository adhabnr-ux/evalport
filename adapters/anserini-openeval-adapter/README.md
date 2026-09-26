# anserini-openeval-adapter

Convert [Anserini](https://github.com/castorini/anserini) (Castorini) TREC
run/qrels output to and from [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM/IR evaluation datasets.

## Origin

Raised as an Idea against Anserini itself
([castorini/anserini#3448](https://github.com/castorini/anserini/discussions/3448),
[castorini/anserini#3449](https://github.com/castorini/anserini/discussions/3449)).
Anserini's maintainer confirmed the adapter belongs here, not in Anserini:

> I'm happy for others to build on Anserini, but this should live in EvalPort.

## Why a standalone package with zero Anserini dependency?

Anserini is a Java toolkit; its actual interchange point with the rest of
the IR ecosystem is the run file, not a Python object. Read directly from
`io/anserini/search/RunOutputWriter.java`, `SearchCollection` writes one of
two formats:

- Standard TREC run format: `qid Q0 docid rank score runtag` (one line per
  retrieved document).
- The `msmarco` variant: `qid<TAB>docid<TAB>rank` (no score/runtag column).

This package works entirely from those two public, stable text formats —
plus a standard TREC qrels file (`qid iteration docid relevance`) — so it
has **zero import dependency on Anserini**: no JVM, no Anserini install, no
Python bindings required. Same "parse the interchange format, don't import
the framework" design `autogen-openeval-adapter` and
`pyserini-openeval-adapter` already use in this repo.

## On IR metrics — honest scope

Anserini itself does not compute effectiveness metrics (MAP, nDCG, recall)
from a run+qrels pair — that has always been `trec_eval`'s job downstream,
and `SearchCollection` only ever writes the ranked list. This adapter fills
that gap with its own small, pure-Python, dependency-free implementations
of MAP, nDCG@k, Recall@k, and MRR, so `to_openeval()` produces a real,
scored `ResultSet` without requiring a JDK, a bundled `trec_eval` binary, or
`pytrec_eval`.

These are standard, documented formulas (linear-gain nDCG with a
`log2(rank + 1)` discount; per-relevant-rank average precision; qrels-count
recall; reciprocal rank of the first relevant hit) — **not** a
reimplementation of `trec_eval`'s C source, and this package makes no claim
to reproduce `trec_eval`'s binary output bit-for-bit (`trec_eval` itself has
used more than one nDCG gain-function convention over the years). Every
`grader_result` is tagged `type: "custom"` with
`params.handler = "anserini_openeval_adapter:<metric>"` precisely so a
consumer that needs `trec_eval`-exact numbers knows to treat this as this
adapter's own scoring, not `trec_eval`'s — the same "type openness" escape
hatch the EvalPort spec defines for exactly this situation.

If you need `trec_eval`-exact scores instead, compute them with
`pyserini.eval.trec_eval` (or a JDK-based `trec_eval` install) and hand the
result to this repo's own
[`pyserini-openeval-adapter`](../pyserini-openeval-adapter), which converts
`trec_eval`'s own per-query output.

## Install

```bash
pip install "anserini-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/anserini-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support (verified working).

## Usage

```python
from anserini_openeval_adapter import to_openeval, from_openeval
from openeval.validate import validate_suite, validate_result_set

# run.trec and qrels.txt are files Anserini's SearchCollection already wrote.
suite, result_set = to_openeval(
    "run.trec",
    "qrels.txt",
    topics_file="topics.tsv",   # optional: qid<TAB>query text, one per line
    suite_id="msmarco-passage-dev",
)

assert validate_suite(suite).valid
assert validate_result_set(result_set).valid

import json
with open("suite.json", "w") as f:
    json.dump(suite, f, indent=2)
with open("results.json", "w") as f:
    json.dump(result_set, f, indent=2)
```

By default, three metrics are scored per query: `map`, `ndcg_cut_10`, and
`recall_100`. Pass your own list of metric names to `metrics=`:

```python
suite, result_set = to_openeval(
    "run.trec", "qrels.txt",
    metrics=("map", "mrr", "ndcg_cut_10", "ndcg_cut_20", "recall_1000"),
)
```

Supported metric names: `"map"`, `"mrr"`, `"ndcg_cut_K"`, `"recall_K"` for
any integer `K`.

### Topics file format

`topics_file` is a simple two-column `qid<TAB>query text` file (the
simplified format used across the IR ecosystem, e.g. BEIR's `queries.tsv`)
— not Anserini/TREC's native SGML/XML topics format. If all you have is a
native topics file, extract each topic's `<title>` field into this format
first. When `topics_file` is omitted, `TestCase.input` falls back to the
query id itself.

### `from_openeval()`

Reconstructs TREC standard run-file text from an EvalPort suite's
`retrieval_context` fields — the inverse direction, for handing a suite
another tool produced back to Anserini-adjacent tooling that expects a run
file:

```python
run_text = from_openeval(suite, runtag="my_run")
with open("reconstructed.run", "w") as f:
    f.write(run_text)
```

This is a documented, one-way-lossy round trip: docids and their relative
order survive; the original retrieval scores and run tag do not (EvalPort's
schema doesn't carry a retrieval score field) — reconstructed lines use a
synthetic descending score of `1 / rank`.

## What round-trips, and what doesn't

Docids, their rank order, and query ids round-trip exactly through
`retrieval_context`. Original retrieval scores, the run tag, and any
non-title topic fields (description/narrative) do not survive — EvalPort's
`TestCase`/`Result` schema has no slot for them. If you need the original
scores preserved, keep the source run file alongside the generated suite
rather than relying on `from_openeval()` to reproduce it exactly.

## Spec

See the full EvalPort specification at
<https://github.com/adhabnr-ux/evalport/blob/main/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
