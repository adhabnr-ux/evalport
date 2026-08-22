# galileo-openeval-adapter

Converts between [Galileo](https://github.com/rungalileo/galileo-python)'s `Dataset` content
rows and real, locally-scored `Trace`/`Span` objects, and
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable
LLM evaluation test cases, graders, suites, and results.

```bash
pip install galileo-openeval-adapter          # adapter only
pip install "galileo-openeval-adapter[galileo]"  # + the real galileo SDK
```

> **Which Galileo package?** This adapter targets `pip install galileo` (the current
> `rungalileo/galileo-python` client, actively released — 2.6.0 as of this writing). Galileo
> also publishes an older, separate `promptquality` package with a completely different, flat
> `PromptRow`-based data model — this adapter does **not** cover that one. If you're on
> `promptquality`, its `PromptRow`/`Metrics` shape is different enough (no trace/span concept,
> a single flat table with `prompt`/`response`/`target`/`metrics` columns) that it would need
> its own adapter, not a compatibility shim bolted onto this one.

```python
from galileo import Dataset, LlmSpan, LocalMetric
from galileo_openeval_adapter import to_openeval, from_openeval, spans_to_openeval

content = [
    {"input": "What is the capital of France?", "output": "Paris"},
    {"input": "What is the capital of Japan?", "output": "Tokyo"},
]

# Before running: describe the dataset as an EvalPort suite
suite = to_openeval(content, suite_id="geo_quiz", ids=["q1", "q2"])

# Restore Dataset-ready rows (e.g. to build a real galileo.Dataset)
rows = from_openeval(suite)
dataset = Dataset(name="geo-quiz", content=rows)

# Run your app, log real Trace/Span objects, score them with a real LocalMetric
spans = [LlmSpan(input=r["input"], output=my_app(r["input"])) for r in rows]

def contains_expected_city(trace_or_span):
    return 1.0 if "paris" in trace_or_span.output.content.lower() else 0.0

metric = LocalMetric(name="contains_expected_city", scorer_fn=contains_expected_city)

# After running: describe the outputs + grades as an EvalPort ResultSet
result_set = spans_to_openeval(
    spans, metrics=[metric], suite_id="geo_quiz", run_id="run-1", ids=["q1", "q2"],
)
```

## Why this exists as a standalone package

`rungalileo/galileo-python` has no `CONTRIBUTING.md` stating a preferred integration path, and
Galileo's SDK is fundamentally a client for a hosted SaaS product — dataset storage,
experiment execution, and every built-in scorer all require a live, authenticated API call.
That's a meaningfully different shape than most frameworks in this ecosystem, and it means
this adapter can only exercise, and can only honestly claim to test, the parts of the SDK that
work fully offline (see "What this adapter does NOT cover" below) — a standalone package makes
that boundary explicit rather than implying deeper integration than actually exists. It
follows the same "standalone package, zero footprint on the target framework" shape as the
AutoGen, CrewAI, Giskard, and Guardrails adapters in this ecosystem.

## Mapping

Galileo's `Dataset` content row is a fully schema-free `dict[str, Any]` — verified by reading
`galileo/dataset.py`'s real `Dataset.__init__` directly: no key names are required or reserved
client-side, `"input"`/`"output"` is a documented convention from the class's own docstring
examples, not an enforced schema.

| Galileo (`Dataset` content row) | EvalPort (`TestCase`) | Notes |
|---|---|---|
| `input_key` (default `"input"`) | `input` | configurable column name |
| `expected_output_key` (default `"output"`) | `expected_output` | configurable column name |
| every other key in the row | `metadata["galileo"]["row"]` | the full original row, for a lossless round trip |

On the results side, `spans_to_openeval()` calls each `LocalMetric`'s real `scorer_fn` against
a real `Trace`/`Span` you constructed (or that Galileo's own logger produced), and turns each
metric's return value into one EvalPort `GraderResult`:

| Galileo (`LocalMetric.scorer_fn` return) | EvalPort (`GraderResult`) |
|---|---|
| the metric's `name` | `grader_id` (slug-normalized) |
| a numeric return value | `score` (clamped to `[0, 1]`) |
| a non-numeric return value (`str`/`list`/`dict`) | `score: null`, raw value in `metadata["raw_value"]` |
| the `(score, metadata)` tuple's second element | `metadata["scorer_metadata"]` |
| `trace_or_span.output`, flattened to text | `actual_output` |
| `trace_or_span.user_metadata` / `.dataset_metadata` / `.id` | `metadata["galileo"]` |

## What this adapter does NOT cover, and why

Galileo's SDK exposes four metric classes — `LocalMetric`, `GalileoMetric`, `LlmMetric`,
`CodeMetric` — but only `LocalMetric` is scored by a plain Python callable this adapter can
call directly and offline. The other three are executed **server-side** against Galileo's
hosted scoring service once a `LogStream`/`Experiment` actually runs; there is no local method
that produces a real score for them without an authenticated network call this adapter cannot
make. `spans_to_openeval()` raises `ValueError` immediately if a non-`LocalMetric` is passed —
per this project's rule against fabricating results, it will not silently skip it or invent a
placeholder score.

Likewise, `Dataset.get_content()`/`.create()` and `Experiment` run results both require a live
API call to produce real data. `to_openeval()`/`from_openeval()` work purely on the `content`
list you already have in hand; `spans_to_openeval()` works on `Trace`/`Span` objects you
construct or already have — neither function triggers a live Galileo call itself.

## Design decisions, documented honestly

**Why every test case references one placeholder `custom` grader in `to_openeval()`.** Galileo
metrics are attached to a `LogStream`/`Experiment` at run time (`log_stream.set_metrics([...])`),
not to a dataset row up front — which metrics will score a given row isn't known until someone
configures a run. The suite-side grader is explicitly labeled a placeholder; the real,
per-metric grading shows up honestly in `spans_to_openeval()`'s output once metrics have
actually scored real spans.

**Why non-numeric `LocalMetric` scores become `score: null`, not a fabricated number.**
`LocalMetric.scorer_fn`'s real return type, read from `galileo/metric.py`, is
`MetricValueType | tuple[MetricValueType, dict]` where `MetricValueType = Union[float, int,
str, None, List[...], Dict[str, ...]]` — a scorer can legitimately return a categorical label
("correct"/"incorrect") or a structured value, not just a float. EvalPort's schema requires
`score` to be a number in `[0, 1]` or `null`, so a non-numeric return honestly becomes
`score: null` with the raw value preserved verbatim in `metadata["raw_value"]`, rather than
inventing a 0/1 mapping this adapter has no principled basis for.

**Two real, non-obvious gotchas found by testing against the installed SDK, not the docs:**

1. **`LlmSpan.output` always coerces a plain string into a real `Message(content=..., role=
   assistant)` object at construction time** — `LlmSpan(output="hello")` does *not* leave
   `.output` as the string `"hello"`; it becomes a `Message`. `Trace.output`, constructed the
   same way, stays a plain string. A `LocalMetric.scorer_fn` written against one Span type and
   assumed to work identically against the other (e.g. `len(trace_or_span.output)`) will break
   on `LlmSpan` with a `TypeError` — this adapter's own `_extract_text()` helper handles both
   shapes, and is worth reusing in your own scorer functions for the same reason.
2. **`LlmSpan.output` rejects a list outright** — `"output must be a Message, a string, or a
   dict"`, confirmed by triggering the real validation error. Only `.input` accepts a list (for
   multi-turn conversation history); `Trace.output`'s list variant is a different, multimodal
   `TextContentPart`/`FileContentPart` shape, not a list of chat messages either. Practically:
   a genuine LLM span always has exactly one output message, never a list of them —
   `_extract_text()`'s list-flattening branch exists for other Span subtypes or plain
   dict/test-double objects that might carry a list-typed `output`, not for `LlmSpan`/`Trace`
   themselves.

## What round-trips losslessly, and what doesn't

Round-trips cleanly: every column in a `Dataset` content row, including columns this adapter
doesn't otherwise interpret (`metadata["galileo"]["row"]` stores the row byte-for-byte).

Does **not** round-trip losslessly:
- **A `LocalMetric`'s non-numeric score loses its type distinction** once converted — a string
  label, a list, and a dict all become `score: null` + `metadata["raw_value"]`; `from_openeval()`
  has no result-side counterpart (`spans_to_openeval()` is one-directional, matching
  `test_results_to_openeval()`-style functions elsewhere in this ecosystem — a "grade" isn't
  something you reconstruct a scorer function from).
- **A `Trace`/`Span`'s full field set beyond `output`/`user_metadata`/`dataset_metadata`/`id`**
  (timing, token counts, nested child spans, tool calls) isn't read by `spans_to_openeval()` —
  it converts scoring results, not a full trace export. If you need the full trace preserved,
  keep it alongside the `ResultSet` rather than expecting it to round-trip through this adapter.
- **A `GalileoMetric`/`LlmMetric`/`CodeMetric` score** can't be produced by this adapter at
  all (see above) — score those through Galileo's own hosted pipeline and, if you want them in
  EvalPort form too, feed the resulting numbers through `spans_to_openeval()` yourself with a
  `LocalMetric` wrapper that just returns the number you already have.

## Testing

47 tests in `tests/test_adapter.py`, all passing against the real, installed `galileo==2.6.0`
package (`Dataset`, `Trace`, `LlmSpan`, `Message`, `MessageRole`, `LocalMetric` are imported
and constructed from the top-level `galileo` package directly, not reinvented — including one
test that trips a real `GalileoMetric` object against `spans_to_openeval()`'s rejection path,
and two tests that pin the exact `Message`-coercion and list-rejection behaviors described
above by triggering them for real) and the real `openeval.validate.validate_suite()` /
`validate_result_set()`. Covers: full field mapping, arbitrary-column preservation, custom
input/expected-output column names, multi-turn list input, invalid input-type rejection,
numeric and non-numeric `LocalMetric` scores, tuple `(score, metadata)` returns, score
clamping at both bounds, custom pass thresholds, multiple metrics per span, `Trace` vs.
`LlmSpan` handling, metadata preservation, summary statistics, and a full suite → `Dataset` →
simulated app run → real `LocalMetric` scoring → `ResultSet` round trip validated end-to-end
against the real spec.

```bash
pip install -e ".[test]"
pip install -e /path/to/evalport/sdk/python   # or: pip install evalport-sdk
pip install galileo==2.6.0
pytest tests/
```
