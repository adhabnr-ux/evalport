---
name: OpenEval Integration Proposal
about: Propose adding OpenEval import/export support to this framework
title: "[Proposal] OpenEval Import/Export Support"
labels: ["enhancement", "interop"]
---

## Proposal

I'd like to propose adding OpenEval import/export support to this framework.

[OpenEval](https://github.com/openeval/openeval) is an open specification (Apache 2.0) for portable LLM evaluation test cases, graders, suites, and results. It defines a standard JSON format that enables eval datasets to be shared across frameworks.

## Why This Benefits Your Users

- **Import eval datasets from other tools** — users on DeepEval, Promptfoo, Inspect AI, LangSmith, etc. can bring their existing eval suites to this framework
- **Export for sharing** — users can export their eval suites to share with teams using other tools
- **Benchmark portability** — community-published benchmarks in OpenEval format work with your framework

## Proposed Implementation

Minimal: add two functions/methods:

```python
# Python
from openeval.convert import from_openeval, to_openeval

def from_openeval(path: str) -> FrameworkTestSuite:
    """Import an OpenEval suite."""
    ...

def to_openeval(suite: FrameworkTestSuite) -> dict:
    """Export to OpenEval format."""
    ...
```

```typescript
// TypeScript
import { fromOpenEval, toOpenEval } from "@openeval/sdk";
```

## What's Already Built

- ✅ Full spec: https://github.com/openeval/openeval/blob/main/spec/SPEC.md
- ✅ TypeScript SDK: `@openeval/sdk` on npm
- ✅ Python SDK: `openeval` on PyPI
- ✅ CLI: `openeval convert promptfoo openeval config.json output.json`
- ✅ Converters for Promptfoo, DeepEval, Inspect AI, OpenAI Evals
- ✅ JSON Schemas for validation

## Offer

I'm happy to submit a PR with a minimal implementation (import/export + tests). No governance or dependency requirements — OpenEval is just a data format.

Would this be useful for your users? Happy to discuss the approach.
