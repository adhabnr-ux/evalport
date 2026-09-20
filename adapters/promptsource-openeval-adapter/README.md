# promptsource-openeval-adapter

Convert [promptsource](https://github.com/bigscience-workshop/promptsource)
(BigScience P3) applied prompt templates to
[EvalPort](https://github.com/adhabnr-ux/evalport), the open interchange
format for portable LLM evaluation datasets.

## Why duck-typed instead of a real dependency?

promptsource's own PyPI release is pinned to Python 3.7–3.9 and hasn't been
re-released for newer interpreters, so it can't be installed alongside a
modern `evalport-sdk` toolchain in one environment. This adapter works
against promptsource's real, documented interface instead of importing the
package: `Template.apply(example)` (returns `[prompt, target]`),
`Template.get_answer_choices_list(example)`, and
`Template.metadata.metrics` — read directly from promptsource's actual
`promptsource/templates.py` source, the same pattern
`autogen-openeval-adapter` already uses for a framework it doesn't import
either.

Proposed and confirmed with promptsource's maintainer first:
[bigscience-workshop/promptsource#860](https://github.com/bigscience-workshop/promptsource/issues/860)
("no changes to PromptSource are necessary if the adapter is in the
EvalPort repo").

## Install

```bash
pip install "promptsource-openeval-adapter @ git+https://github.com/adhabnr-ux/evalport.git#subdirectory=adapters/promptsource-openeval-adapter"
```

Not yet published to PyPI — this installs directly from source via pip's
`git+`/`#subdirectory=` support (verified working).

## Usage

### One template across a dataset split

```python
from promptsource.templates import DatasetTemplates
from datasets import load_dataset
from promptsource_openeval_adapter import template_to_openeval
from openeval.validate import validate_suite

templates = DatasetTemplates("ag_news")
template = templates["classify_question_first"]
examples = list(load_dataset("ag_news", split="test"))

suite = template_to_openeval(template, examples, dataset_name="ag_news")
assert validate_suite(suite).valid
```

promptsource's own contract (its CONTRIBUTING.md, quoted in the proposal
issue): "a template must produce two strings: an input and a target",
separated by `|||`. The rendered prompt string becomes `TestCase.input`,
the rendered target string becomes `expected_output`, and rendered
`answer_choices` (when the template has any) are preserved under
`test_case.metadata.promptsource.answer_choices`.

### A whole `DatasetTemplates` collection at once

```python
from promptsource_openeval_adapter import dataset_templates_to_openeval

suite = dataset_templates_to_openeval(templates, examples, dataset_name="ag_news")
# one test case per (template, example) pair
```

Pass `template_names=[...]` to apply only a subset of the collection's
templates. Graders shared by name across templates (e.g. two templates
both tagged `Accuracy`) are deduplicated into a single `Suite.graders`
entry rather than duplicated.

## Grader inference from `metadata.metrics`

Only `"Accuracy"` — one of promptsource's own fixed metric-tag vocabulary —
is automatically typed as EvalPort's `"exact_match"` grader; that type has
no required `params`, so the mapping can't misrepresent anything. Every
other tag (`"BLEU"`, `"ROUGE"`, `"Squad"`, `"Pearson Correlation"`,
`"Mean Reciprocal Rank"`, ...) maps to EvalPort's `"custom"` grader type
with `params.handler` set to the metric name, since EvalPort's typed
options (`llm_judge`, `semantic_similarity`) each require real
configuration (a model + prompt, a similarity threshold) this module has
no honest value for. A template with no `metadata.metrics` at all falls
back to a plain `exact_match` grader rather than leaving test cases with no
matching grader.

## `from_openeval()`

A live promptsource `Template` can't be reconstructed from an EvalPort
suite (its Jinja source isn't part of an applied prompt's output), so
`from_openeval()` returns the flattened record shape this adapter itself
produces instead — `{input, target, answer_choices, dataset, subset,
template_name}` per test case.

## Testing note

promptsource's Python-version pin means its real package can't run in this
adapter's own CI. The test suite exercises the exact, real Jinja template
string and `answer_choices` expression from promptsource's own
`ag_news/templates.yaml` (quoted verbatim in the proposal issue above),
rendered through the real `jinja2` package using logic copied from
promptsource's actual `Template.apply()`/`get_answer_choices_list()` — real
template rendering, not a hand-typed fixture standing in for one.

## Spec

See the full EvalPort specification at
<https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md>.

## License

Apache 2.0 — see [LICENSE](LICENSE).
