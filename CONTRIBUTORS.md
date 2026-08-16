# Contributors

EvalPort is a specification, and specifications only matter if real implementations adopt them. This file credits people who have built real, working EvalPort integrations outside this repository — not just filed issues or left comments, but shipped code against the spec.

## Framework Integration Authors

| Contributor | Contribution |
|---|---|
| [SparshGarg999](https://github.com/SparshGarg999) | Built native OpenEval dataset import/export helpers directly into `openai/openai-python` ([PR #3619](https://github.com/openai/openai-python/pull/3619), closing [#3549](https://github.com/openai/openai-python/issues/3549)) — `to_openeval()`/`from_openeval()` conforming to `spec/schemas/testcase.json`, with lossless round-tripping of multi-turn conversations, tool calls, and multimodal content preserved under `metadata`. |
| [DresdenGman](https://github.com/DresdenGman) | Built the AutoGen OpenEval adapter ([PR #8009](https://github.com/microsoft/autogen/pull/8009), closing [#8005](https://github.com/microsoft/autogen/issues/8005)) — `autogenstudio.eval.openeval` with `to_openeval()`/`from_openeval()`, migrated cleanly to the renamed `evalport-sdk` distribution when the project renamed from OpenEval, and kept backward-compatible imports for existing users. |

## Maintainer Reviewers

Maintainers who evaluated EvalPort integration proposals on their own merits and gave substantive technical feedback (scope, semver policy, lossiness handling) rather than a rubber-stamp:

| Reviewer | Repository | Contribution |
|---|---|---|
| [Josh Reini](https://github.com/sfc-gh-jreini) | [truera/trulens](https://github.com/truera/trulens) | Scoped and approved the `to_openeval()`/`from_openeval()` design in [#2680](https://github.com/truera/trulens/issues/2680) before the PR was opened. |
| [Charles Teague](https://github.com/dragonstyle) | [UKGovernmentBEIS/inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) | Reviewed, iterated on, and merged EvalPort into Inspect AI's official community extensions list ([PR #4797](https://github.com/UKGovernmentBEIS/inspect_ai/pull/4797)). |

---

Want to add a framework integration or improve the spec itself? See [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md).
