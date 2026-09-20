"""Tests for promptsource_openeval_adapter.

promptsource's own PyPI release is pinned to Python 3.7-3.9 and can't be
installed alongside a modern evalport-sdk toolchain in this environment, so
this module duck-types against promptsource's real interface rather than
importing the package (see the module docstring). To still exercise real
behavior rather than a hand-typed fixture pretending to be one, `_Template`
below is a faithful, minimal re-implementation of promptsource's actual
`Template`/`Template.Metadata` classes -- `apply()` and
`get_answer_choices_list()` are copied near-verbatim from promptsource's
real `promptsource/templates.py` (Apache 2.0, same license as this
package), including its pipe-escaping logic for the `|||` separator. The
Jinja template string and answer_choices expression used in the fixtures
below are the *real*, unmodified content of promptsource's own
`ag_news/templates.yaml`, quoted verbatim in
bigscience-workshop/promptsource#860 -- rendered through the real `jinja2`
package, not asserted as pre-computed output. Every conversion is validated
against the real evalport-sdk (`openeval.validate.validate_suite`).
"""
from __future__ import annotations

import pytest
from jinja2 import BaseLoader, Environment

from promptsource_openeval_adapter import (
    dataset_templates_to_openeval,
    from_openeval,
    template_to_openeval,
)
from openeval.validate import validate_suite

_env = Environment(loader=BaseLoader)
_PIPE_PROTECTOR = "3ed2dface8203c4c9dfb1a5dc58e41e0"


def _escape_pipe(example):
    return {
        k: v.replace("|||", _PIPE_PROTECTOR) if isinstance(v, str) else v
        for k, v in example.items()
    }


def _unescape_pipe(s):
    return s.replace(_PIPE_PROTECTOR, "|||")


class _Metadata:
    """Faithful to promptsource's real `Template.Metadata`."""

    def __init__(self, metrics=None):
        self.metrics = metrics


class _Template:
    """Faithful, minimal re-implementation of promptsource's real
    `Template` class (`promptsource/templates.py`), reproducing its actual
    `apply()`/`get_answer_choices_list()` rendering logic (Apache 2.0) so
    tests exercise genuine Jinja rendering rather than a canned string."""

    def __init__(self, name, jinja, answer_choices=None, metrics=None):
        self.name = name
        self.jinja = jinja
        self.answer_choices = answer_choices
        self.metadata = _Metadata(metrics=metrics)

    def get_name(self):
        return self.name

    def get_answer_choices_list(self, example):
        if self.answer_choices is None:
            return None
        rtemplate = _env.from_string(self.answer_choices)
        protected = _escape_pipe(example)
        rendered = rtemplate.render(**protected)
        return [_unescape_pipe(c.strip()) for c in rendered.split("|||")]

    def apply(self, example):
        rtemplate = _env.from_string(self.jinja)
        protected = _escape_pipe(example)
        protected["answer_choices"] = self.get_answer_choices_list(example)
        rendered = rtemplate.render(**protected)
        return [_unescape_pipe(part).strip() for part in rendered.split("|||")]


class _DatasetTemplates:
    """Faithful to promptsource's real `DatasetTemplates.__getitem__` /
    `.all_template_names` interface."""

    def __init__(self, templates):
        self._by_name = {t.name: t for t in templates}

    @property
    def all_template_names(self):
        return sorted(self._by_name.keys())

    def __getitem__(self, name):
        return self._by_name[name]


# ---------------------------------------------------------------------------
# Fixtures -- the REAL ag_news/templates.yaml content quoted verbatim in
# bigscience-workshop/promptsource#860.
# ---------------------------------------------------------------------------

_AG_NEWS_JINJA = (
    "What label best describes this news article?\n"
    "{{text}} ||| \n"
    "{{answer_choices[label] }}"
)
_AG_NEWS_ANSWER_CHOICES = "World politics ||| Sports ||| Business ||| Science and technology"


@pytest.fixture
def ag_news_template():
    return _Template(
        name="classify_question_first",
        jinja=_AG_NEWS_JINJA,
        answer_choices=_AG_NEWS_ANSWER_CHOICES,
        metrics=["Accuracy"],
    )


@pytest.fixture
def ag_news_examples():
    # Real ag_news schema: {"text": str, "label": int in [0,3]}.
    return [
        {"text": "Stocks rallied on Friday after strong earnings.", "label": 2},
        {"text": "The national team won the championship match.", "label": 1},
        {"text": "Researchers unveil a new quantum computing chip.", "label": 3},
    ]


class TestTemplateToOpenEval:
    def test_real_jinja_rendering_and_validates(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(
            ag_news_template, ag_news_examples, dataset_name="ag_news"
        )
        validation = validate_suite(suite)
        assert validation.valid, validation.errors
        assert len(suite["test_cases"]) == 3

    def test_rendered_input_matches_real_jinja_output(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(ag_news_template, ag_news_examples, dataset_name="ag_news")
        tc0 = suite["test_cases"][0]
        assert tc0["input"] == (
            "What label best describes this news article?\n"
            "Stocks rallied on Friday after strong earnings."
        )

    def test_expected_output_is_rendered_answer_choice(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(ag_news_template, ag_news_examples, dataset_name="ag_news")
        # label=2 -> answer_choices[2] == "Business"
        assert suite["test_cases"][0]["expected_output"] == "Business"
        # label=1 -> "Sports"
        assert suite["test_cases"][1]["expected_output"] == "Sports"
        # label=3 -> "Science and technology"
        assert suite["test_cases"][2]["expected_output"] == "Science and technology"

    def test_answer_choices_preserved_in_metadata(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(ag_news_template, ag_news_examples, dataset_name="ag_news")
        assert suite["test_cases"][0]["metadata"]["promptsource"]["answer_choices"] == [
            "World politics",
            "Sports",
            "Business",
            "Science and technology",
        ]

    def test_accuracy_metric_maps_to_exact_match_grader(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(ag_news_template, ag_news_examples, dataset_name="ag_news")
        assert suite["graders"] == [{"id": "Accuracy", "type": "exact_match", "description": (
            "promptsource template metadata.metrics entry 'Accuracy', "
            "inferred as EvalPort's 'exact_match' grader (no additional "
            "params required)."
        )}]
        for tc in suite["test_cases"]:
            assert tc["graders"] == ["Accuracy"]

    def test_unrecognized_metric_falls_back_to_custom(self, ag_news_examples):
        template = _Template(
            name="rouge_template", jinja="{{text}} ||| {{text}}", metrics=["ROUGE"]
        )
        suite = template_to_openeval(template, ag_news_examples, dataset_name="ag_news")
        grader = suite["graders"][0]
        assert grader["type"] == "custom"
        assert grader["params"] == {"handler": "ROUGE"}
        assert validate_suite(suite).valid

    def test_no_metrics_falls_back_to_plain_exact_match(self, ag_news_examples):
        template = _Template(name="open_ended", jinja="{{text}} ||| {{text}}", metrics=None)
        suite = template_to_openeval(template, ag_news_examples, dataset_name="ag_news")
        assert suite["graders"] == [{"id": "exact_match", "type": "exact_match"}]
        assert validate_suite(suite).valid

    def test_no_answer_choices_omits_metadata_key(self, ag_news_examples):
        template = _Template(name="open_ended", jinja="{{text}} ||| summary", metrics=["Accuracy"])
        suite = template_to_openeval(template, ag_news_examples, dataset_name="ag_news")
        assert "answer_choices" not in suite["test_cases"][0]["metadata"]["promptsource"]

    def test_default_suite_id_includes_dataset_and_template(
        self, ag_news_template, ag_news_examples
    ):
        suite = template_to_openeval(ag_news_template, ag_news_examples, dataset_name="ag_news")
        assert suite["id"] == "promptsource_ag_news_classify_question_first"

    def test_subset_name_included_in_default_ids(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(
            ag_news_template, ag_news_examples, dataset_name="glue", subset_name="mrpc"
        )
        assert suite["id"] == "promptsource_glue_mrpc_classify_question_first"
        assert suite["test_cases"][0]["id"] == "glue_mrpc_classify_question_first_0"

    def test_custom_ids(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(
            ag_news_template, ag_news_examples, dataset_name="ag_news", ids=["a", "b", "c"]
        )
        assert [tc["id"] for tc in suite["test_cases"]] == ["a", "b", "c"]

    def test_dataset_and_template_recorded_in_metadata(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(ag_news_template, ag_news_examples, dataset_name="ag_news")
        meta = suite["test_cases"][0]["metadata"]["promptsource"]
        assert meta["dataset"] == "ag_news"
        assert meta["subset"] is None
        assert meta["template_name"] == "classify_question_first"

    def test_empty_examples_raises(self, ag_news_template):
        with pytest.raises(ValueError, match="examples is empty"):
            template_to_openeval(ag_news_template, [], dataset_name="ag_news")

    def test_mismatched_ids_length_raises(self, ag_news_template, ag_news_examples):
        with pytest.raises(ValueError, match="ids has length"):
            template_to_openeval(
                ag_news_template, ag_news_examples, dataset_name="ag_news", ids=["only_one"]
            )

    def test_non_two_part_apply_raises(self, ag_news_examples):
        # A template missing the "|||" separator entirely violates
        # promptsource's own documented contract.
        bad_template = _Template(name="broken", jinja="{{text}} no separator here")
        with pytest.raises(ValueError, match="must return exactly"):
            template_to_openeval(bad_template, ag_news_examples, dataset_name="ag_news")


class TestDatasetTemplatesToOpenEval:
    def test_merges_multiple_templates_into_one_suite(self, ag_news_examples):
        t1 = _Template(name="t1", jinja="{{text}} ||| a", metrics=["Accuracy"])
        t2 = _Template(name="t2", jinja="{{text}} ||| b", metrics=["Accuracy"])
        collection = _DatasetTemplates([t1, t2])
        suite = dataset_templates_to_openeval(collection, ag_news_examples, dataset_name="ag_news")
        assert validate_suite(suite).valid
        # 2 templates * 3 examples = 6 test cases
        assert len(suite["test_cases"]) == 6

    def test_deduplicates_shared_grader_ids(self, ag_news_examples):
        t1 = _Template(name="t1", jinja="{{text}} ||| a", metrics=["Accuracy"])
        t2 = _Template(name="t2", jinja="{{text}} ||| b", metrics=["Accuracy"])
        collection = _DatasetTemplates([t1, t2])
        suite = dataset_templates_to_openeval(collection, ag_news_examples, dataset_name="ag_news")
        # Both templates use "Accuracy" -- must appear once, not twice
        # (validate_suite() would flag a literal duplicate as DUPLICATE_ID).
        assert [g["id"] for g in suite["graders"]] == ["Accuracy"]

    def test_selects_only_named_templates(self, ag_news_examples):
        t1 = _Template(name="t1", jinja="{{text}} ||| a", metrics=["Accuracy"])
        t2 = _Template(name="t2", jinja="{{text}} ||| b", metrics=["Accuracy"])
        collection = _DatasetTemplates([t1, t2])
        suite = dataset_templates_to_openeval(
            collection, ag_news_examples, dataset_name="ag_news", template_names=["t1"]
        )
        assert len(suite["test_cases"]) == 3
        assert all(
            tc["metadata"]["promptsource"]["template_name"] == "t1" for tc in suite["test_cases"]
        )

    def test_unknown_template_name_raises(self, ag_news_examples):
        t1 = _Template(name="t1", jinja="{{text}} ||| a", metrics=["Accuracy"])
        collection = _DatasetTemplates([t1])
        with pytest.raises(ValueError, match="not found in dataset_templates"):
            dataset_templates_to_openeval(
                collection, ag_news_examples, dataset_name="ag_news", template_names=["nope"]
            )

    def test_empty_template_names_raises(self, ag_news_examples):
        t1 = _Template(name="t1", jinja="{{text}} ||| a", metrics=["Accuracy"])
        collection = _DatasetTemplates([t1])
        with pytest.raises(ValueError, match="no templates to apply"):
            dataset_templates_to_openeval(
                collection, ag_news_examples, dataset_name="ag_news", template_names=[]
            )

    def test_real_ag_news_template_end_to_end(self, ag_news_template, ag_news_examples):
        collection = _DatasetTemplates([ag_news_template])
        suite = dataset_templates_to_openeval(collection, ag_news_examples, dataset_name="ag_news")
        assert validate_suite(suite).valid
        assert suite["test_cases"][0]["expected_output"] == "Business"


class TestFromOpenEval:
    def test_round_trip_records(self, ag_news_template, ag_news_examples):
        suite = template_to_openeval(ag_news_template, ag_news_examples, dataset_name="ag_news")
        records = from_openeval(suite)
        assert len(records) == 3
        assert records[0]["input"] == suite["test_cases"][0]["input"]
        assert records[0]["target"] == "Business"
        assert records[0]["answer_choices"] == [
            "World politics",
            "Sports",
            "Business",
            "Science and technology",
        ]
        assert records[0]["dataset"] == "ag_news"
        assert records[0]["template_name"] == "classify_question_first"

    def test_empty_suite_raises(self):
        with pytest.raises(ValueError, match="no test_cases"):
            from_openeval({"version": "1.0.0", "id": "s", "graders": [], "test_cases": []})
