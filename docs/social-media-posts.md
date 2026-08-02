# Social Media Posts — Manual Posting Required

*No API access for HN, Reddit, Dev.to, or LinkedIn. Copy/paste these manually.*

---

## Hacker News

**Title:** EvalPort: Why LLM Evaluation Needs a Standard Format
**URL:** https://github.com/adhabnr-ux/evalport

*Post at 9 AM PT / 12 PM ET on a weekday. Engage with every comment in the first 2 hours.*

---

## Reddit — r/MachineLearning

**Title:** [P] EvalPort: A portable standard for LLM evaluation datasets

**Body:**
We just open-sourced EvalPort, an open standard for representing evaluation test cases, graders, and results. The problem: every major eval framework (DeepEval, Promptfoo, Inspect AI, LangSmith, etc.) uses incompatible formats. You can't share eval datasets across tools.

EvalPort solves this with:
- Portable JSON format for test cases, graders, and results
- 11 grader types (exact_match, semantic_similarity, llm_judge, etc.)
- Converters for Promptfoo, DeepEval, Inspect AI, OpenAI Evals
- TypeScript + Python SDKs
- CLI for validation and conversion

Spec: https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md
Repo: https://github.com/adhabnr-ux/evalport
Try it: `pip install openeval` or `npm install @evalport/sdk`

We've filed issues on 17 eval framework repos proposing import/export support. Would love feedback from the community.

---

## Reddit — r/LocalLLaMA

**Title:** EvalPort: Run the same eval suite in DeepEval, Promptfoo, Inspect AI — portable eval standard just shipped

**Body:**
Same as above.

---

## Reddit — r/ArtificialIntelligence

**Title:** EvalPort v1.0.0-rc.1: The Open LLM Evaluation Standard

**Body:**
Same as above.

---

## Dev.to

**Frontmatter:**
```
---
title: "EvalPort: Why LLM Evaluation Needs a Standard Format"
description: "An open standard for portable LLM evaluation datasets and results"
tags: llm, evaluation, ai, testing
---
```

**Body:** Copy from docs/blog/launch-post.md

---

## LinkedIn

**Post:**
Excited to announce EvalPort v1.0.0-rc.1! 🚀

After months of work, we're shipping an open standard for LLM evaluation datasets. The problem: every eval framework (DeepEval, Promptfoo, Inspect AI, LangSmith, OpenAI Evals) uses incompatible formats. You can't share eval datasets across tools.

EvalPort solves this with:
✅ Portable JSON format for test cases, graders, and results
✅ TypeScript + Python SDKs
✅ Converters for Promptfoo, DeepEval, Inspect AI, OpenAI Evals
✅ CLI for validation and conversion
✅ Full spec + examples + migration guides
✅ 11 grader types covering 90%+ of eval needs

Try it:
npm install @evalport/sdk
pip install openeval

Repo: https://github.com/adhabnr-ux/evalport
Spec: https://github.com/adhabnr-ux/evalport/blob/main/spec/SPEC.md

Would love feedback from the eval community!

Tags: #LLM #AI #Evaluation #OpenSource #StandardsForAI
