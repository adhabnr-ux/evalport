# pyserini-openeval-adapter

Convert [pyserini](https://github.com/castorini/pyserini) (Castorini)
`trec_eval()` information-retrieval evaluation results to and from
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM/IR evaluation datasets.

## Why a standalone package with zero pyserini dependency?

`pyserini.eval.trec_eval.trec_eval()` isn't a pure-Python function — read
directly from its real source (`pyserini/eval/trec_eval.py`), it shells out
to a bundled `trec_eval` Java binary via `jnius` and a JVM. That means
requiring `pyserini` itself here would force a JDK plus a full pyserini
install onto everyone who wants to convert a `trec_eval()` result, just to
read the plain Python `dict` it already returns.

So this adapter has **zero import dependency on pyserini**: it works
entirely from `trec_eval(..., return_per_query_results=True)`'s own return
shape — a `dict` keyed by query id (plus the aggregate `"all"` key), e.g.
`{"301": 0.4231, "302": 0.5510, "all": 0.4823}` — the same pattern
`autogen-openeval-adapter` already uses for a framework it doesn't want to
import either.

Proposed and confirmed with pyserini's maintainer first:
[castorini/pyserini#2651](https://github.com/castorini/pyserini/issues/2651)
("you can implement the adaptor on the evalport end... doesn't require
anything from us").

## A real footgun this adapter is built around

`trec_eval.py`'s own source has a `# TODO: FIXME` comment: because the
returned dict is keyed by query id rather than by metric, requesting
multiple `-m` metrics in a single `trec_eval()` call makes each later
metric's per-query values silently overwrite the earlier ones for the same
query id — only the *last* requested metric's numbers actually survive.
This adapter doesn't paper over that; it's built to require you call
`trec_eval()` once per metric and hand every metric's dict to it together
(see "Multiple metrics" below), so nothing gets silently dropped.

## Install

```bash
pip install "pyserini-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/pyserini-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support (verified working).

## Usage

### Single metric

```python
from pyserini.eval.trec_eval import trec_eval
from pyserini_openeval_adapter import results_to_openeval
from openeval.validate import validate_result_set

per_query = trec_eval(
    ['-m', 'ndcg_cut.10', qrels_file, run_file],
    return_per_query_results=True,
)
# per_query: {"301": 0.4231, "302": 0.5510, ..., "all": 0.4823}

result_set = results_to_openeval(
    per_query,
    metric="ndcg_cut_10",
    suite_id="beir-arguana-test",
)
assert validate_result_set(result_set).valid
```

### Multiple metrics — the correct way, per the FIXME above

```python
per_query_ndcg = trec_eval(['-m', 'ndcg_cut.10', qrels_file, run_file],
                            return_per_query_results=True)
per_query_map = trec_eval(['-m', 'map', qrels_file, run_file],
                           return_per_query_results=True)

result_set = results_to_openeval(
    None,
    metrics={"ndcg_cut_10": per_query_ndcg, "map": per_query_map},
    suite_id="beir-arguana-test",
)
```

Each query now carries one `grader_result` per metric, rather than only the
last metric's value.

### `passed` and `pass_threshold`

IR metrics have no universal "good" value — nDCG@10 of 0.4 might be strong
on a hard dataset and weak on an easy one. By default, a (query, metric)
pair "passes" when its value is `> 0.0` (any non-zero relevance signal).
Pass `pass_threshold=...` explicitly with a value meaningful for your
metric and collection; don't rely on the default for anything but a basic
non-zero check. A query's overall `passed` is the AND of every metric's
`passed` for that query.

### Score clamping

A few trec_eval pseudo-metrics (`num_ret`, `num_rel`, `num_rel_ret`, ...)
are raw counts, not `[0, 1]` fractions. Values outside `[0, 1]` are clamped
to fit EvalPort's required range, with the unclamped raw value preserved in
`grader_result.metadata.pyserini.raw_score` whenever clamping changed
anything.

### `from_openeval()`

Converts an EvalPort ResultSet back into a trec_eval-shaped
`{query_id: value, "all": aggregate}` dict — the inverse of the
single-metric form, useful for handing a ResultSet another tool produced
back to pyserini-based reporting code. Pass `metric=...` to disambiguate
when a Result carries more than one grader result.

## What round-trips, and what doesn't

Per-query scores round-trip exactly. `ResultSet.summary.metadata.pyserini
.trec_eval_aggregate` preserves trec_eval's own `"all"` value per metric
verbatim; `from_openeval()`'s own `"all"` key is instead the mean of the
*extracted* per-query scores (EvalPort's schema doesn't require the
original aggregate to survive a round trip) — prefer the summary field if
you need the exact original trec_eval aggregate back.

## Spec

See the full EvalPort specification at
<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
