---
name: EvalPort Integration Proposal
about: Propose adding EvalPort import/export support to this framework
title: "[Proposal] EvalPort Import/Export Support"
labels: ["enhancement", "interop"]
---

## Proposal

I'd like to propose adding EvalPort import/export support to this framework.

[EvalPort](https://github.com/adhabnr-ux/evalport) is an open specification (Apache 2.0) for portable LLM evaluation test cases, graders, suites, and results. It defines a standard JSON format that enables eval datasets to be shared across frameworks.

## Why This Benefits Your Users

- **Import eval datasets from other tools** — users on DeepEval, Promptfoo, Inspect AI, LangSmith, etc. can bring their existing eval suites to this framework
- **Export for sharing** — users can export their eval suites to share with teams using other tools
- **Benchmark portability** — community-published benchmarks in EvalPort format work with your framework

## Proposed Implementation

Minimal: add two functions/methods:

```python
# Python
from openeval.convert import from_openeval, to_openeval

def from_openeval(path: str) -> FrameworkTestSuite:
    """Import an EvalPort suite."""
    ...

def to_openeval(suite: FrameworkTestSuite) -> dict:
    """Export to EvalPort format."""
    ...
```

```typescript
// TypeScript
import { fromEvalPort, toEvalPort } from "evalport-sdk"; // npm install evalport-sdk
```

## What's Already Built

- ✅ Full spec: https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md
- ✅ TypeScript SDK: `evalport-sdk` on npm
- ✅ Python SDK: `openeval` on PyPI
- ✅ CLI: `openeval convert promptfoo openeval config.json output.json`
- ✅ Converters for Promptfoo, DeepEval, Inspect AI, OpenAI Evals
- ✅ JSON Schemas for validation

## Offer

I'm happy to submit a PR with a minimal implementation (import/export + tests). No governance or dependency requirements — EvalPort is just a data format.

Would this be useful for your users? Happy to discuss the approach.
