# EvalPort → Every Eval Ever (EEE)

> **Status: unsolicited prototype.** This code has not been proposed as a pull
> request to, reviewed by, or accepted by the Every Eval Ever maintainers. It is
> the concrete version of the question in
> [evaleval/every_eval_ever#262](https://github.com/evaleval/every_eval_ever/issues/262),
> which had no replies when this was written. EEE's `CONTRIBUTING.md` asks for the
> approach to be agreed with a maintainer before a structural change is opened as a
> PR, and this repository's code does not follow EEE's in-tree converter
> conventions (pydantic types, the registry-resolved metric ids, `ConverterCase`
> tests, `uv` in docs). Treat it as a worked example of the mapping, not as
> something EEE has endorsed or will take as is.

[Every Eval Ever](https://github.com/evaleval/every_eval_ever) stores aggregate
evaluation results per model, described by `eval.schema.json` (schema version
`0.3.0` here). `evalport_to_eee.py` turns one EvalPort Suite plus one ResultSet into
**one EEE aggregate record**, always with `source_metadata.source_type:
"evaluation_run"`. It is one-way and lossy, and needs only the standard library.

EvalPort does not carry several things EEE requires, namely the model identity,
whether the model is self-deployed, whether its weights are open, who is providing
the data, and the evaluator's relationship to the model developer. They are
required command-line options, and the converter fails rather than guessing them.

```bash
python evalport_to_eee.py ../basic-suite.json ../results.json \
    --model-id openai/gpt-4o \
    --deployment-type externally_managed \
    --model-availability closed_weights \
    --source-organization-name "Example Org" \
    --evaluator-relationship third_party \
    -o record.json
```

`python evalport_to_eee.py --help` lists the optional options (model name,
developer, inference platform, dataset URL or Hugging Face repo, eval-library
overrides, a fixed `--retrieved-timestamp`). Exit status is 0 on success, 1 on a
conversion error (message on stderr, nothing on stdout), and 2 on a usage error
such as a missing required option.

## Aggregation

One EEE record per ResultSet. Inside it:

- **One result per grader**: the **mean of that grader's non-null scores**.
  `evaluation_result_id` is `<suite_id>/<grader_id>/mean_score`.
- **One suite-level result**, `<suite_id>/test_case_pass_rate`: the share of test
  cases whose `passed` is true, over cases that have at least one non-null grader
  score. It is tagged `aggregation_level: suite` (graders are `grader`) in
  `metric_config.additional_details`. For a suite with a single grader the two
  numbers are usually close or equal, so a consumer should not add them together.
- **Null scores** (`score: null`, "not verified", SPEC Validation Rule 6) are left
  out of every mean and every rate. They are never counted as 0 and never as a
  failure. EEE's `score_details.score` is a required number with no null, so the
  exclusions are recorded as strings in `score_details.details`
  (`scored_count`, `unscored_count`, `passed_count`, `pass_rate_over_scored`; for
  the suite result `scored_cases`, `unscored_cases`, `passed_cases`).
- A grader with **no** non-null score gets **no** evaluation result at all, and is
  named in `source_metadata.additional_details.graders_without_scored_results`.
  If nothing in the run is scored, the converter refuses to write an empty record.
- **Uncertainty** is written only when it can be computed honestly: the standard
  error of the mean, `sample SD (n − 1) / √n`, plus the SD and `num_samples`
  (`method: "analytic"`). It is left out when there are fewer than two scored
  results, or when any test case has repeated attempts, because attempts of the
  same case are not independent items and the simple formula would understate the
  error. The reason is written to `details.uncertainty_omitted`. No confidence
  interval is computed. A sample of identical scores gives a standard error of 0;
  that is what the formula says, not a claim of certainty.

## Mapping

| EvalPort | EEE (`eval.schema.json` 0.3.0) |
|---|---|
| (caller) `--model-id` | `model_info.id`; `name` defaults to `provider.model`, else the id |
| (caller) `--deployment-type`, `--model-availability` | `model_info.additional_details.deployment_type` / `model_availability` (required by EEE's validator) |
| (caller) `--developer`, `--inference-platform` | `model_info.developer`, `inference_platform` (only when given; `developer` is never derived from the id) |
| `provider.model`, when different from `--model-id` | `model_info.additional_details.evalport_provider_model` |
| (caller) `--source-organization-name`, `--evaluator-relationship`, `--source-organization-url` | `source_metadata.*` |
| always | `source_metadata.source_type: "evaluation_run"` |
| `runner.name` | `source_metadata.source_name` (else `EvalPort`) and `eval_library.name` (else `unknown`) |
| `runner.version` | `eval_library.version` (else `unknown`) |
| `ResultSet.started_at` (needs a UTC offset) | `evaluation_timestamp` (Unix epoch seconds as a string) and the last part of `evaluation_id` |
| `Suite.id`, `model_id`, `started_at` | `evaluation_id` = `<suite_id>/<model_id>/<epoch>`, stable across re-conversion |
| (now, or `--retrieved-timestamp`) | `retrieved_timestamp` |
| `Suite.id` | `evaluation_results[].evaluation_name` |
| `Suite.name` (else id), `suite_version`, test case count; `--dataset-url` / `--hf-repo` / `--hf-split` | `evaluation_results[].source_data` (`other` by default, `url` or `hf_dataset` when given) |
| `grader_results[].grader_id`, `type` | `metric_config.metric_id` = `evalport.grader.<id>`, `metric_name`, `additional_details.evalport_grader_*` |
| `Suite.graders[].description` | appended to `metric_config.evaluation_description` |
| `grader_results[].score` (non-null) | `score_details.score` (mean), `uncertainty`, `details` |
| `grader_results[].passed` | `details.passed_count`, `details.pass_rate_over_scored` |
| `Result.passed` | the suite-level `test_case_pass_rate` result |
| `provider.temperature`, `max_tokens` | `generation_config.generation_args` |
| `provider.extra` | `generation_config.additional_details` (`provider_extra.<key>`, values JSON-encoded) |
| `run_id`, `suite_id`, `version`, result count, null and error counts | `source_metadata.additional_details` |

Every `metric_config` is `continuous`, `min_score: 0.0`, `max_score: 1.0`,
`lower_is_better: false`: EvalPort scores are null or in [0, 1] (SPEC Validation
Rule 5) and pass/fail is decided by comparing the score to a threshold (SPEC, Validation Rules).

## What is lost

- **Per-test-case results.** Inputs, expected and actual outputs, per-case scores,
  `reason`, `duration_ms`, `attempt` numbering, `completed_at` and result
  `metadata` are not exported. EEE keeps these in a separate instance-level
  `_samples.jsonl`, which this converter does not write. Mapping them would hit the
  same null problem again: the instance-level `evaluation.score` is a required
  number and `is_correct` a required boolean, so a not-verified row has nowhere to go.
  The aggregate counts (`evalport_result_count`, `unscored_*`) are all that is left.
- **`score: null` as a value.** It cannot be stored. It survives only as a count in
  a string map and, for a grader with no scores, as a name in a list. A consumer who
  reads `score_details.score` alone sees a mean over fewer items than the suite has
  and nothing in the typed fields says so. (EEE's `levels` metrics have a
  `has_unknown_level` switch under which a score of -1 means unknown. This
  converter does not use it: its metrics are continuous means, not levels.)
- **Grader definitions.** `params` (thresholds, patterns, expected values, code),
  `weight`, and the `metadata.openeval.aggregation` strategy (`any`, `majority`,
  `weighted`) are not exported, and a grader's weight is not applied to any mean.
  `Result.passed` is used as the producer wrote it. For `llm_judge` graders the judge
  model and prompt are not exported either: EEE's `llm_scoring` requires a full
  `model_info` for each judge (including its deployment axes), which EvalPort does
  not hold.
- **Other ResultSet fields.** `summary` is ignored and recomputed from `results`.
  `isolation`, `group`, `completed_at`, `metadata`, and the suite's `config` and
  `tags` are dropped. `provider.api_base` is dropped on purpose (it can be an
  internal endpoint). Error objects are reduced to a count.
- **Canonical identities.** Metric ids are `evalport.*` and marked
  `metric_id_status: unregistered`; they join with nothing else in EEE. The model id
  is whatever `--model-id` says; it is not resolved through EEE's eval-card-registry.
- **Datastore placement.** The script writes a record, not a datastore entry. EEE
  files live at `data/<collection>/<developer>/<model>/<uuid4>.json` in the Hugging
  Face datastore and are submitted through a PR there; none of that is done here.

## Judgement calls

- `evaluation_id` uses the run's `started_at`, not "now", so converting the same run
  twice gives the same id. Two different runs that start in the same second with the
  same model would collide; `run_id` is kept in `source_metadata.additional_details`.
- `source_data` is `other` with the suite name unless a dataset URL or Hugging Face
  repo is given, because an EvalPort suite carries its cases inline and names no
  public dataset.
- `eval_library` comes from `runner` (falling back to `unknown`, the fallback EEE's
  schema describes for the version). `Suite.config.provider` is not used: it is the
  configuration the suite asked for, and `ResultSet.provider` is what ran.
- The suite-level `test_case_pass_rate` result is an addition of this converter, not
  something EEE asks for.
- Test cases in the ResultSet that the suite does not contain, a grader that appears
  with two different types, and scores outside [0, 1] are rejected, not repaired.

## Tests

```bash
pip install -e ../../sdk/python -r requirements.txt
pytest -p no:cacheprovider .
```

`test_evalport_to_eee.py` checks:

1. the conversion: arithmetic (hand-computed means and standard errors), null-score
   handling, uncertainty being left out, suite-id mismatch being rejected, missing or
   invalid required context being rejected, and the CLI (stdout, `-o`, exit codes);
2. that every produced record validates against the vendored `eval.schema.json`
   (Draft 7) with `jsonschema`. The example inputs are first checked with
   `openeval.validate`. `FormatChecker` is enabled, though this schema declares no
   `format` keyword, so today it checks nothing extra;
3. negative controls: 18 one-fault mutations of a valid record (a dropped required
   field, a bad enum, a non-string `additional_details` value, a null score, …) must
   each fail that same schema check, which shows it is not a no-op;
4. when `every-eval-ever` is installed (Python 3.12 or newer, see
   `requirements.txt`), the same records go through EEE's own validator
   (`validate_aggregate(..., run_semantic_checks=True)`: pydantic models plus the
   merge-gate checks for score bounds, deployment axes and model identity path), with
   its own negative controls. On older Pythons those tests are reported as skipped.

CI runs these tests in `.github/workflows/every-eval-ever-example.yml` on Python 3.12.
`vendor/eval.schema.json` is upstream's file, unmodified (commit and checksum in
`vendor/README.md`); one test compares it with the schema inside the installed
`every-eval-ever` package, so upstream drift shows up when the pin is bumped.
