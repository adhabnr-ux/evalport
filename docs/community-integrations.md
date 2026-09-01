# Community Integrations

This page tracks real-world projects that have built their **own** native support
for EvalPort's test-case format, independently of this repository.

This is a different thing from [`adapters/`](../adapters/): those packages are
`to_openeval()`/`from_openeval()` converters that EvalPort itself authors and
maintains, one per framework. The entries below are integrations written and
owned by the other project, in their own codebase, that happen to speak
EvalPort's field names. They're listed here for visibility, not bundled as
EvalPort packages.

## ragbits-evaluate (deepsense-ai/ragbits)

[`ragbits`](https://github.com/deepsense-ai/ragbits) is deepsense-ai's toolkit
for building GenAI applications, and `ragbits-evaluate` is its evaluation
package. Its `DataLoader` base class
(`packages/ragbits-evaluate/src/ragbits/evaluate/dataloaders/base.py`) wraps
`datasets.load_dataset(...)` with a `required_keys` check, and
`QuestionAnswerDataLoader`
(`packages/ragbits-evaluate/src/ragbits/evaluate/dataloaders/question_answer.py`)
maps arbitrary dataset column names into a `QuestionAnswerData(question,
reference_answer, reference_context)` record — a shape that lines up closely
with an EvalPort [`TestCase`](../spec/schemas/testcase.json): `input` →
`question`, `expected_output` → `reference_answer`, `retrieval_context` →
`reference_context`.

Following up on a proposal in
[deepsense-ai/ragbits#986](https://github.com/deepsense-ai/ragbits/issues/986),
ragbits maintainer [@mikemikimike](https://github.com/mikemikimike) implemented
an `EvalPortDataLoader` in
[deepsense-ai/ragbits#989](https://github.com/deepsense-ai/ragbits/pull/989):

```python
class EvalPortDataLoader(QuestionAnswerDataLoader):
    """Load EvalPort test cases as question-answer evaluation data."""

    def __init__(self, source: Source, *, split: str = "data") -> None:
        super().__init__(
            source=source,
            split=split,
            question_key="input",
            answer_key="expected_output",
            context_key="retrieval_context",
        )
```

It subclasses `QuestionAnswerDataLoader` directly, defaulting to EvalPort's own
field names (`input`, `expected_output`, `retrieval_context`) and a `"data"`
split, so an EvalPort suite file loads through the existing question-answer
evaluation path without the caller having to pass
`question_key=`/`answer_key=`/`context_key=` by hand. `required_keys` stays
scoped to `{"input", "expected_output"}`, so `retrieval_context` remains
optional, and the class is exported from `ragbits.evaluate.dataloaders`
alongside the existing loaders. It's covered by
`packages/ragbits-evaluate/tests/unit/test_dataloaders.py`, which checks both
the field mapping and the `split`/`required_keys` defaults.

As of this writing, PR #989 is open on `deepsense-ai/ragbits` and pending
review — it has not yet merged. It's linked here as a genuine, independently
written example of a project mapping its own data loader onto EvalPort's
test-case field names; this entry will be updated if the PR's status changes.

---

Know of another project with a genuine, working EvalPort integration? Open a
PR adding it here, or start a
[Discussion](https://github.com/adhabnr-ux/evalport/discussions).
