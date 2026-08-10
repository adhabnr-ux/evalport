# SQuAD 2.0 — Reading Comprehension

500 cases converted from the SQuAD 2.0 validation split into a validated
EvalPort suite.

## Source

- **Dataset**: [rajpurkar/squad_v2](https://huggingface.co/datasets/rajpurkar/squad_v2) (`validation` split)
- **Original site**: [rajpurkar.github.io/SQuAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/)
- **License**: CC-BY-SA-4.0 — see [../LICENSES.md](../LICENSES.md) (share-alike: redistributing this suite further requires carrying the same attribution)
- **Paper**: Rajpurkar, Jia & Liang, 2018 — [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/abs/1806.03822)

## What's in the suite

Reading-comprehension questions over Wikipedia passages, including SQuAD
2.0's defining feature: a substantial share of questions (roughly half in
this suite — 263 of the first 500 validation rows) are deliberately
**unanswerable** from the given passage, testing whether a model correctly
declines to answer rather than hallucinating a plausible-sounding one. The
prompt explicitly instructs the model to respond "unanswerable" when the
passage doesn't contain the answer; `expected_output` is either the first
gold answer span or the literal string `"unanswerable"`.

## Grader rationale

Per-case inline `contains` graders (substring match against the expected
answer, `ignore_case: true`) rather than a shared suite-level grader —
each case's expected substring is unique, so it has to live inline (see
[`../humaneval/README.md`](../humaneval/README.md) for the general
explanation of why per-case-varying params require inline graders). This is
explicitly a lightweight proxy for SQuAD's official token-level F1/Exact
Match metric, which this suite does not reproduce — substring containment
will under-count partially-correct answers and can be gamed by a model that
pads its answer with extra text containing the right substring. Treat
scores from this suite as directional (see
`metadata.evalport.grading_note` in the suite file).

## Case count

500 (of 11,873 available in the validation split — capped by the pipeline's
default `--limit 500`; includes 263 unanswerable questions in this
particular 500-row slice).

## Quickstart

```bash
cd ../_tools && python3 validate_all.py ../squad2/squad2.json
```
