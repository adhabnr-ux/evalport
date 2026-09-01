
# Contributors

EvalPort is a specification, and specifications only matter if real implementations adopt them. This file credits people who have built real, working EvalPort integrations — whether that's a native integration shipped in another project's own repository, or a standalone adapter package shipped inside this repo's `adapters/` directory — not just filed issues or left comments, but shipped code against the spec — plus maintainers whose review substantively shaped the spec or an adapter, whether the answer was yes or not yet.

## Framework Integration Authors

| Contributor | Contribution |
|---|---|
| [SparshGarg999](https://github.com/SparshGarg999) | Built native OpenEval dataset import/export helpers directly into `openai/openai-python` ([PR #3619](https://github.com/openai/openai-python/pull/3619), closing [#3549](https://github.com/openai/openai-python/issues/3549)) — `to_openeval()`/`from_openeval()` conforming to `spec/schemas/testcase.json`, with lossless round-tripping of multi-turn conversations, tool calls, and multimodal content preserved under `metadata`. |
| [DresdenGman](https://github.com/DresdenGman) | Built the AutoGen OpenEval adapter ([PR #8009](https://github.com/microsoft/autogen/pull/8009), closing [#8005](https://github.com/microsoft/autogen/issues/8005)) — `autogenstudio.eval.openeval` with `to_openeval()`/`from_openeval()`, migrated cleanly to the renamed `evalport-sdk` distribution when the project renamed from OpenEval, and kept backward-compatible imports for existing users. |

## Adapter Package Contributors

People who built a standalone `to_openeval()`/`from_openeval()` package that ships inside this repo's own `adapters/` directory — real, tested, merged code, not just a request or an idea:

| Contributor | Contribution |
|---|---|
| [VimalN2005](https://github.com/VimalN2005) | Built [`parea-openeval-adapter`](adapters/parea-openeval-adapter/) ([PR #19](https://github.com/adhabnr-ux/evalport/pull/19), closing [#17](https://github.com/adhabnr-ux/evalport/issues/17)) — their first contribution to the project. Addressed a full round of review feedback (score clamping, an accidentally-committed lockfile, an empty-name fallback, a numeric-id cast fix) correctly on the first pass, with real test coverage added for each fix. Followed up with [`humanloop-openeval-adapter`](adapters/humanloop-openeval-adapter/) ([PR #26](https://github.com/adhabnr-ux/evalport/pull/26), closing [#15](https://github.com/adhabnr-ux/evalport/issues/15)) — a clean second contribution handling Humanloop's dual `inputs`/`messages` datapoint shapes and all four `EvaluatorReturnTypeEnum` judgment types (boolean/numeric/select/text) honestly, merged after independent verification against the real `humanloop` SDK and EvalPort spec validators. |
| [Sidd-1507](https://github.com/Sidd-1507) | Built [`literalai-openeval-adapter`](adapters/literalai-openeval-adapter/) ([PR #25](https://github.com/adhabnr-ux/evalport/pull/25), closing [#23](https://github.com/adhabnr-ux/evalport/issues/23)) — handling Literal AI's schema-free dict-shaped `input`/`expected_output`, unbounded `Score.value` clamping into `[0, 1]` with the raw value preserved in `metadata.openeval.raw_score`, and the `HUMAN`/`CODE`/`AI` grader-type mapping. Took a full round of schema-conformance feedback (the initial output didn't validate against `spec/schemas/suite.json`/`resultset.json`) and turned it around correctly, with 38 tests passing against the real `openeval.validate` validators and the real `literalai` package by the second pass. |

## Maintainer Reviewers

Maintainers who evaluated EvalPort integration proposals on their own merits and gave substantive technical feedback (scope, semver policy, lossiness handling, version-pinning gaps) rather than a rubber-stamp — credited here whether their answer was yes, not yet, or "fix this first":

| Reviewer | Repository | Contribution |
|---|---|---|
| [Josh Reini](https://github.com/sfc-gh-jreini) | [truera/trulens](https://github.com/truera/trulens) | Scoped and approved the `to_openeval()`/`from_openeval()` design in [#2680](https://github.com/truera/trulens/issues/2680) before the PR was opened, then reviewed and approved the resulting implementation in [PR #2697](https://github.com/truera/trulens/pull/2697) ("Solid implementation... 17 comprehensive tests validating against real spec validators"). |
| [Charles Teague](https://github.com/dragonstyle) | [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | Reviewed, iterated on, and merged EvalPort into Inspect AI's official community extensions list ([PR #4797](https://github.com/UKGovernmentBEIS/inspect_ai/pull/4797)). |
| [marliessophie](https://github.com/marliessophie) | [langfuse/langfuse](https://github.com/langfuse/langfuse) | Gave a considered, specific answer on why a first-party integration wasn't the right call yet (stability and adoption bar) rather than a silent close ([issue #16174](https://github.com/langfuse/langfuse/issues/16174#issuecomment-5329834060)) — exactly the kind of real engagement this list exists to credit, independent of the outcome. |
| [Julian Risch](https://github.com/julian-risch) | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | Engaged with the `haystack-openeval-adapter` grader-mapping proposal in [issue #12361](https://github.com/deepset-ai/haystack/issues/12361), and acknowledged the shipped adapter once it was built ("Glad to hear you went ahead with this and implemented the adapter! 👏"). |

---

Want to add a framework integration or improve the spec itself? See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md).
