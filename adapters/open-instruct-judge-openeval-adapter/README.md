# open-instruct-judge-openeval-adapter

Convert judge score/rationale output from [allenai/open-instruct](https://github.com/allenai/open-instruct)'s
`open_instruct/judge_utils.py` to and from [EvalPort](https://github.com/adhabnr-ux/evalport),
the open interchange format for portable LLM evaluation datasets.

## Why a standalone package?

`extract_json_score_with_fallback()` plus the `JUDGE_PROMPT_MAP` / `EXTRACTOR_MAP`
registry in open-instruct already produce a `(reasoning, score)` pair for every
`{input, output, label}` triple they judge — structurally an EvalPort
`GraderResult` (score + rationale) attached to a `Result`. This package turns a
batch of those judged examples into a real, schema-valid EvalPort `ResultSet`,
so open-instruct's judge output can be compared against another
EvalPort-speaking tool's grading on the same test cases.

It lives here, as a standalone adapter, rather than inside `open_instruct/`
itself, per the maintainer's review in
[open-instruct#1922](https://github.com/allenai/open-instruct/issues/1922):
a prototype here gives something concrete to review before committing to an
in-repo integration, and this placement doesn't require waiting on that
review to land. This does not rule out a future optional export helper
inside `open_instruct/` — see that issue for the ongoing discussion.

**This package never imports `open_instruct`.** That package's own
`__init__.py` pulls in vllm / deepspeed / flash-attn / torch — a full
training+inference stack this adapter has no reason to require just to
reshape already-judged records after the fact. Instead, this adapter is a
self-contained, pure-stdlib reimplementation of the exact parsing behavior
of `extract_json_score_with_fallback()` and `extract_score_web_instruct()`
in `judge_utils.py`, pinned to commit
[`677512f`](https://github.com/allenai/open-instruct/blob/677512ff387914b083a444db2c7a05ca995c114c/open_instruct/judge_utils.py)
(2026-10-05, the version read and reviewed in the issue above). It needs no
judge calls and no training dependencies — only `evalport-sdk`.

## Install

```
pip install "open-instruct-judge-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/open-instruct-judge-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support.

## Usage

Each judged example is a dict (or any attribute-bearing object) describing
one `{input, output, label}` triple that open-instruct's judge scored:

```python
from open_instruct_judge_openeval_adapter import to_openeval, from_openeval
from openeval.validate import validate_result_set

examples = [
    {
        "id": "ex1",
        "judge_type": "quality",  # a JUDGE_PROMPT_MAP / EXTRACTOR_MAP key
        "input": "What is the capital of France?",
        "output": "Paris.",
        "raw_judge_response": '{"REASONING": "Correct and concise.", "SCORE": "8"}',
        "judge_model": "gpt-4o",
    },
]

result_set = to_openeval(examples, suite_id="oi_judge_eval", run_id="run_2026_10_05")

validation = validate_result_set(result_set)
assert validation.valid, validation.errors

import json
with open("my_results.json", "w") as f:
    json.dump(result_set, f, indent=2)

round_tripped = from_openeval(result_set)
```

### Two ways to supply a score

- **`raw_judge_response`** (the judge model's raw string reply): the adapter
  parses and normalizes it itself, exactly mirroring the real extractor for
  that `judge_type` (JSON-first, regex fallback, then "not verified" — see
  below).
- **`score`** (a float you already produced, e.g. by calling the real
  `EXTRACTOR_MAP[judge_type]` yourself): trusted as-is and **never
  re-normalized**. This matters specifically for `quality` / `quality_ref`,
  whose real extractor already divides by 10 — passing a pre-divided score
  through `raw_judge_response` instead would double-normalize it.

### `judge_type` and score scale

| `judge_type` | Extractor mirrored | Scale |
|---|---|---|
| `quality` | `extract_score_with_fallback_max_10` | raw 1–10, divided by 10 here |
| `quality_ref` | `extract_score_with_fallback_max_10` | raw 1–10, divided by 10 here |
| `quality_rubric` | `extract_json_score_with_fallback` | 0 or 1, used as-is |
| `safety` | `extract_json_score_with_fallback` | used as-is |
| `factuality` | `extract_json_score_with_fallback` | used as-is |
| `creative_writing` | `extract_json_score_with_fallback` | used as-is |
| `refusal` | `extract_json_score_with_fallback` | used as-is |
| `web_instruct_general_verifier` | `extract_score_web_instruct` | binary 0.0/1.0 |

`safety`, `factuality`, `creative_writing` and `refusal`'s templates are
marked TODO/incomplete in open-instruct itself as of the pinned commit, so
their real score range isn't contractually guaranteed to land in `[0, 1]`
the way EvalPort's `GraderResult.score` requires. This adapter clips
defensively to `[0, 1]` rather than letting one out-of-range score reject an
entire batch, and sets `grader_results[].metadata.score_clipped = True` when
it does, so the clip is visible rather than silent.

### Parse failures are not silently zero

open-instruct's own extractors return `0.0` whenever a judge response fails
to parse — the same value a genuine, correctly-judged zero would produce.
That ambiguity is a real data-quality hazard for anything downstream that
treats `0.0` as a meaningful grade, so this adapter does not reproduce it:

- A judge response that parses — whether cleanly (`parse_status: "json"`)
  or via the same regex fallback the real extractor uses
  (`parse_status: "regex_fallback"`) — gets its real, possibly-zero score.
- A judge response that parses in neither way gets `score: None` and
  `passed: False` (`parse_status: "failed"`), matching EvalPort's
  `SPEC.md` Validation Rule 6: `score: null` means "not verified." The raw
  response is kept in `grader_results[].metadata.raw_judge_response` either
  way, so a failure can still be inspected by a human or re-judged later —
  nothing is thrown away.

Do not treat every `score: 0.0` you see from open-instruct's own raw judge
logs as equivalent to this adapter's `0.0` unless it went through this same
re-parsing: upstream's own logs can't make that distinction after the fact
(see the parsing-ambiguity discussion in
[open-instruct#1922](https://github.com/allenai/open-instruct/issues/1922)).

One faithfully-mirrored upstream quirk worth knowing: a ` ```json ` fence
on its own line (with the JSON body starting on the *next* line) leaves a
literal leading `\n` immediately before `{` once newlines are escaped,
which is not valid at the start of a JSON document — `json.loads()`
genuinely fails there in the real `extract_json_score_with_fallback()`
too, not just here, and falls through to the regex fallback (which still
recovers the real score). See
`tests/test_adapter.py::test_fenced_multiline_json_falls_back_to_regex`.

### Batches with more than one judge type

A single call to `to_openeval()` accepts a batch mixing any of the eight
`judge_type`s above — each becomes one `Result` with one `GraderResult`
(`grader_id: "open_instruct_<judge_type>"`), all sharing one `suite_id` /
`run_id`. See `tests/test_adapter.py::test_batch_with_multiple_judge_types_validates_as_one_result_set`
for a worked example spanning five judge types in one validated `ResultSet`.

### Stable case IDs and provenance

Pass `id` (or `case_id`) on each example for a stable `test_case_id`; without
one, the adapter falls back to a batch-order `tc_<index>` id. `input`,
`output` and `label` are preserved on the output `Result` untouched (`input`
and `label` under `metadata`, `output` as `actual_output`), alongside
whatever of `judge_model`, `prompt_key`, `extractor_name`, `version`/`commit`
and a free-form `metadata` dict you supply — so a `ResultSet` this adapter
produces is enough, on its own, to trace a score back to the judge call,
prompt/registry key, extractor and code version that produced it.

## Credit

This adapter's mapping follows the requirements set out by `@abhishekraok`
in review of the original proposal, in
[open-instruct#1922](https://github.com/allenai/open-instruct/issues/1922#issuecomment-6000667122):
stable case IDs with provenance, exact-once score normalization per
`judge_type`, and a parse-failure representation that never masquerades as
a genuine zero.

## Spec

See the full EvalPort specification at
https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

## License

Apache 2.0 — see LICENSE.
