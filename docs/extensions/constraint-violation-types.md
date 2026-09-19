# Extensions Registry: `constraint_violations[].type` starter vocabulary

> **Status:** non-normative. This list is a *seed*, not an enumeration. `constraint_violations[].type`
> is an open string by design (see [`SPEC.md`'s Hard Constraints section](../../SPEC.md), from
> [Discussion #47](https://github.com/adhabnr-ux/evalport/discussions/47)) -- new categories can be
> used freely without a spec revision or a change to this file. This document exists so producers
> converge on the same spelling for common cases instead of each inventing their own, which is the
> entire value of `type` being RECOMMENDED rather than absent: it is the one field on a
> `constraint_violations` entry that can carry meaning *across* suites, and it can only do that if
> independent producers land on the same string for the same concept.
>
> This file is the repo-tracked source for the registry entry described in `SPEC.md`'s Extensions
> Registry section (published at `https://evalport.org/extensions`). To propose an addition, open a
> PR against this file with a real producer's usage as the motivating example -- the same standard
> every other RFC in this project holds itself to (see `CONTRIBUTORS.md`).

| `type` | Meaning | Motivating example |
|---|---|---|
| `authorization` | The result depended on an action or data access the requesting role/identity was not permitted to perform. | AOBench's RBAC hard fail ([Discussion #47](https://github.com/adhabnr-ux/evalport/discussions/47)): an agent answers correctly using data outside its role's scope. |
| `safety` | The result violates a safety policy (harmful content, disallowed instructions followed, etc.), independent of whether the *content* was otherwise high quality. | A model that produces a fluent, on-topic, but policy-violating response -- high `grader_results` scores on relevance/fluency, but the response should never have been produced. |
| `privacy` | The result leaked or mishandled personally identifiable or otherwise sensitive information. | A summarization task whose output correctly compresses a document but includes a PII field the task explicitly required be redacted. |
| `licensing` | The result depended on content whose license/terms prohibit the way it was used or reproduced. | A retrieval-augmented answer that verbatim-quotes a source under a license that disallows redistribution at that length. |
| `resource_policy` | The result violated a quota, budget, or scheduling-class constraint rather than a content or access rule. | An HPC/agent-ops task where an agent's correct answer required exceeding its allotted compute/cost budget or violating a scheduling-class restriction -- the case that motivated adding this category, since it does not fit `authorization`/`safety`/`privacy`/`licensing` (raised during [MSKazemi/aobench#51](https://github.com/MSKazemi/aobench/discussions/51#discussioncomment-18405164)'s review of this RFC). |

## Adding a category

1. Open a PR against this file.
2. Add one row: the proposed `type` string, its meaning in one sentence, and a real (not
   hypothetical) example of a producer that needs it -- an actual benchmark, harness, or eval
   framework, the same evidentiary bar `spec/SPEC.md`'s other RFCs hold themselves to.
3. No schema change, no SDK change, and no spec-version bump are needed -- that is the point of
   `type` being an open string with a non-normative registry rather than a closed enum.
