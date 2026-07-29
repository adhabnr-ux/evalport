# Founder Outreach Emails — Manual Send Required

*Gmail auth expired. Copy/paste these manually.*

---

## Email 1: Jeffrey Ip (DeepEval)

**To:** Find on GitHub/LinkedIn
**Subject:** OpenEval + DeepEval — Portable Eval Datasets

Hi Jeffrey,

We just launched OpenEval (v1.0.0-rc.1), an open standard for LLM evaluation datasets.

The problem: Eval datasets are locked into each framework. A test suite built in DeepEval can't be run in Promptfoo without manual conversion.

We've built OpenEval with a portable JSON format + converters + SDKs. We've already built a DeepEval converter (maps FaithfulnessMetric, AnswerRelevancyMetric, etc. to OpenEval graders).

I filed an issue on the DeepEval repo proposing import/export support. Would you have 15 minutes to discuss?

Spec: https://github.com/adhabnr-ux/openeval/blob/main/spec/SPEC.md
GitHub: https://github.com/adhabnr-ux/openeval

Best,
Sahi

---

## Email 2: Michael D'Amour (Promptfoo)

**Subject:** OpenEval + Promptfoo — Portable Eval Datasets

Hi Michael,

We just launched OpenEval, an open standard for LLM evaluation datasets.

The problem: Eval datasets are locked into each framework. A test suite built in Promptfoo can't be run in DeepEval without manual conversion.

We've built a Promptfoo → OpenEval converter already. Your users could import eval suites from other tools and export theirs for sharing.

I filed an issue on the Promptfoo repo. Happy to submit a PR with the implementation if you're interested.

GitHub: https://github.com/adhabnr-ux/openeval
Migration guide: https://github.com/adhabnr-ux/openeval/blob/main/docs/migration-guides/promptfoo-to-openeval.md

Best,
Sahi

---

## Email 3: UK AISI (Inspect AI)

**Subject:** OpenEval + Inspect AI — Portable Eval Datasets

Hi AISI team,

We just launched OpenEval, an open standard for LLM evaluation datasets.

Government eval standards align with OpenEval's reproducibility goals. We've built an Inspect AI converter (maps Sample/target to TestCase, scorers to graders).

I filed an issue on the inspect_ai repo. Would love to collaborate on native OpenEval support.

GitHub: https://github.com/adhabnr-ux/openeval
Migration guide: https://github.com/adhabnr-ux/openeval/blob/main/docs/migration-guides/inspect-to-openeval.md

Best,
Sahi

---

## Email 4: LangChain Team

**Subject:** OpenEval + LangSmith — Portable Eval Datasets

Hi LangChain team,

We just launched OpenEval, an open standard for LLM evaluation datasets.

LangSmith datasets are already JSON — OpenEval makes them portable to other tools. Your users could import eval suites from DeepEval, Promptfoo, etc.

I filed an issue on the langsmith-sdk repo.

GitHub: https://github.com/adhabnr-ux/openeval

Best,
Sahi

---

## Email 5: EleutherAI

**Subject:** OpenEval + lm-evaluation-harness — Portable Benchmark Results

Hi EleutherAI team,

We just launched OpenEval, an open standard for LLM evaluation datasets.

lm-evaluation-harness results need a standard format for cross-model comparison. OpenEval provides a portable ResultSet format that any tool can consume.

I filed an issue on the lm-evaluation-harness repo.

GitHub: https://github.com/adhabnr-ux/openeval

Best,
Sahi
