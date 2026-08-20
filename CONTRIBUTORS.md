# Contributors

EvalPort is a specification, and specifications only matter if real implementations adopt them. This file credits people who have built real, working EvalPort integrations outside this repository — not just filed issues or left comments, but shipped code against the spec — plus maintainers whose review substantively shaped the spec or an adapter, whether the answer was yes or not yet.

## Framework Integration Authors

| Contributor | Contribution |
|---|---|
| [SparshGarg999](https://github.com/SparshGarg999) | Built native OpenEval dataset import/export helpers directly into `openai/openai-python` ([PR #3619](https://github.com/openai/openai-python/pull/3619), closing [#3549](https://github.com/openai/openai-python/issues/3549)) — `to_openeval()`/`from_openeval()` conforming to `spec/schemas/testcase.json`, with lossless round-tripping of multi-turn conversations, tool calls, and multimodal content preserved under `metadata`. |
| [DresdenGman](https://github.com/DresdenGman) | Built the AutoGen OpenEval adapter ([PR #8009](https://github.com/microsoft/autogen/pull/8009), closing [#8005](https://github.com/microsoft/autogen/issues/8005)) — `autogenstudio.eval.openeval` with `to_openeval()`/`from_openeval()`, migrated cleanly to the renamed `evalport-sdk` distribution when the project renamed from OpenEval, and kept backward-compatible imports for existing users. |

## Maintainer Reviewers

Maintainers who evaluated EvalPort integration proposals on their own merits and gave substantive technical feedback (scope, semver policy, lossiness handling, version-pinning gaps) rather than a rubber-stamp — credited here whether their answer was yes, not yet, or "fix this first":

| Reviewer | Repository | Contribution |
|---|---|---|
| [Josh Reini](https://github.com/sfc-gh-jreini) | [truera/trulens](https://github.com/truera/trulens) | Scoped and approved the `to_openeval()`/`from_openeval()` design in [#2680](https://github.com/truera/trulens/issues/2680) before the PR was opened, then reviewed and approved the resulting implementation in [PR #2697](https://github.com/truera/trulens/pull/2697) ("Solid implementation... 17 comprehensive tests validating against real spec validators"). |
| [Charles Teague](https://github.com/dragonstyle) | [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | Reviewed, iterated on, and merged EvalPort into Inspect AI's official community extensions list ([PR #4797](https://github.com/UKGovernmentBEIS/inspect_ai/pull/4797)). |
| [yun520-1](https://github.com/yun520-1) | [run-llama/llama_index](https://github.com/run-llama/llama_index) | Caught a real gap in `llamaindex-openeval-adapter` — the target-framework version constraint lived only in test extras, with no equivalent for a real install ([issue #22709](https://github.com/run-llama/llama_index/issues/22709#issuecomment-5310707592)) — which became [Discussion #13](https://github.com/adhabnr-ux/evalport/discussions/13) and a packaging-convention fix backported across the whole adapter fleet. |
| [marliessophie](https://github.com/marliessophie) | [langfuse/langfuse](https://github.com/langfuse/langfuse) | Gave a considered, specific answer on why a first-party integration wasn't the right call yet (stability and adoption bar) rather than a silent close ([issue #16174](https://github.com/langfuse/langfuse/issues/16174#issuecomment-5329834060)) — exactly the kind of real engagement this list exists to credit, independent of the outcome. |
| [Julian Risch](https://github.com/julian-risch) | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | Engaged with the `haystack-openeval-adapter` grader-mapping proposal in [issue #12361](https://github.com/deepset-ai/haystack/issues/12361), and acknowledged the shipped adapter once it was built ("Glad to hear you went ahead with this and implemented the adapter! 👏"). |

---

Want to add a framework integration or improve the spec itself? See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md).
