# argilla-openeval-adapter

Convert [Argilla](https://github.com/argilla-io/argilla) `Record` objects (fields, suggestions, and human responses) to and from [EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange format for portable LLM evaluation datasets.

## Why Argilla is a genuinely good fit for EvalPort

An Argilla `Record` more or less *is* an evaluation-dataset row: `fields` are the inputs shown to an annotator, `suggestions` are pre-computed candidate judgments (e.g. from an LLM-judge pass run before human review), and `responses` are the real, completed human judgments. That maps almost directly onto EvalPort's `TestCase` (input + graders) and `ResultSet` (grader results) — which is exactly the "portable LLM evaluation dataset" this project exists to standardize.

## Install

```bash
pip install "argilla-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/argilla-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's `git+`/`#subdirectory=` support (verified working).

## A real constraint this adapter works *around*, not into

Confirmed directly against `argilla` 2.8.0: constructing a live `argilla.Settings`, `Field`, `Question`, or `Dataset` object requires a connected `Argilla` client — and that client validates the connection **eagerly**, at construction time (`Argilla.__init__` calls `GET /api/v1/me` before it returns; with no server reachable this raises `httpx.ConnectError`). A conversion library has no business requiring a live server just to translate data between formats, so this adapter only ever touches the parts of the Argilla object model that are constructible fully offline — `Record`, `Suggestion`, and `Response` — all three confirmed instantiable and serializable with zero network calls. Dataset *settings* (field/question definitions) are represented here as plain data (field names, preserved in suite/test-case metadata) rather than live Argilla objects: you build the real `rg.Settings` yourself, once you have a connected client, from the field names this adapter preserves losslessly.

## Usage

### Export Argilla records as a portable EvalPort suite

```python
import argilla as rg
from argilla_openeval_adapter import to_openeval

records = [
    rg.Record(id="q1", fields={"prompt": "What is the capital of Japan?"}),
    rg.Record(id="q2", fields={"prompt": "What is 7 * 6?"}),
]

suite = to_openeval(records, suite_id="geo_math_quiz")

from openeval.validate import validate_suite
assert validate_suite(suite).valid
```

Every test case is graded by a single `"human"` grader (`{"id": "human", "type": "human"}`) — the one EvalPort grader type with zero required `params`, and the honest choice for a platform whose entire purpose is capturing human judgment. Records with more than one field become an array `input` (`["value1", "value2", ...]`), with the original field names preserved losslessly in `metadata.argilla.field_names` so `from_openeval` can reconstruct them.

Any pre-existing `suggestions` on a record — candidate judgments produced *before* human review — are preserved verbatim under `metadata.argilla.suggestions` rather than promoted to executed grader results: a suggestion is a proposal, not yet a validated result, and this adapter never fabricates the latter from the former.

### Import an EvalPort suite as Argilla-ready record specs

```python
from argilla_openeval_adapter import from_openeval

specs = from_openeval(suite)  # -> [{"id": "q1", "fields": {"prompt": "..."}, ...}, ...]
records = [rg.Record.from_dict(spec) for spec in specs]

# once you have a connected client and a dataset with matching Settings:
# dataset.records.log(records)
```

`from_openeval` returns plain dicts shaped exactly like `Record.to_dict()` output — pass each one through `argilla.Record.from_dict()` yourself, which is Argilla's own stable reconstruction path (this adapter doesn't hardcode a `Record(...)` constructor call, so it never silently depends on one installed version's constructor signature).

### Export completed human annotations as an EvalPort ResultSet

```python
import uuid
from argilla_openeval_adapter import responses_to_openeval

records[0].responses.add(
    rg.Response(question_name="correct", value=True, user_id=uuid.uuid4(), status="submitted")
)
records[1].responses.add(
    rg.Response(question_name="correct", value=True, user_id=uuid.uuid4(), status="submitted")
)

result_set = responses_to_openeval(records, ids=["q1", "q2"], suite_id="geo_math_quiz")

from openeval.validate import validate_result_set
assert validate_result_set(result_set).valid
```

Every question a record has at least one `Response` for becomes one `GraderResult` of `type: "human"` — real, completed human judgment, never a fabrication. Argilla allows more than one annotator to answer the same question on the same record; when a question has exactly one response its `grader_id` is the question name, and when it has several each gets its own `grader_id` of `"<question_name>[<index>]"` so no annotator's judgment is silently dropped or averaged away. The responding annotator's real `user_id` and `status` are preserved in each grader result's `metadata`.

### The full loop

```python
suite = to_openeval(records, suite_id="geo_math_quiz")
specs = from_openeval(suite)
live_records = [rg.Record.from_dict(s) for s in specs]
# ... annotators complete their review in a real, connected Argilla dataset ...
live_records[0].responses.add(rg.Response(question_name="correct", value=True, user_id=uuid.uuid4(), status="submitted"))
result_set = responses_to_openeval(live_records, ids=["q1"], suite_id=suite["id"])
```

## Scoring: what gets a number, and what honestly can't

A boolean response value maps to `1.0`/`0.0`. A numeric response value already in `[0, 1]` is used directly. A numeric value outside that range (e.g. a 1-5 `RatingQuestion`) is normalized via an optional `rating_ranges` argument — `responses_to_openeval(records, rating_ranges={"quality": (1, 5)})` — and left as `score: null` when no range is supplied, rather than guessing a scale. A label, multi-label, ranking, span, or free-text response value always has `score: null`: there's no honest way to turn "the annotator picked label X" into a number without knowing what counts as correct, so this adapter doesn't invent one.

`passed` is `score >= passing_threshold` (default `0.5`, overridable) when a score was computed, and `True` otherwise — a completed response with no computable score still represents a human having reviewed and judged the item, which this adapter treats as a pass by default. Override `passing_threshold`, or post-process `passed` yourself for stricter label-matching semantics (e.g. comparing a label response against `TestCase.expected_output`).

Records with no responses at all (not yet annotated) are skipped by `responses_to_openeval` — there's nothing real to report yet. If *no* record in the batch has any responses, it raises `ValueError` rather than returning an empty, meaningless `ResultSet`.

## What round-trips losslessly, and what doesn't

Argilla → EvalPort → Argilla (via this adapter): field values, field names (for multi-field records), record ids, record metadata, and suggestions all round-trip exactly, confirmed by test (`TestFullLoop`).

Argilla → EvalPort → some other tool: the input fields, suggestions, and human judgments are readable by any EvalPort consumer, but a different tool has no way to reconstruct Argilla-specific concepts like question *types* (`LabelQuestion` vs. `RatingQuestion` vs. `SpanQuestion`, etc.) — those live in `Settings`, which (per the constraint above) requires a connected client to construct and isn't something a pure data adapter should fabricate. You define your dataset's real `Settings` once, using the field names this adapter preserves.

## Spec

See the full EvalPort specification at <https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
