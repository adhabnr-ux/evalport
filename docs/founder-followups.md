# Founder Email Follow-ups (3+ days elapsed)

*Send these manually from Gmail — original emails were sent July 29*

> **Archived drafts, dated 2026-07-31. The statuses in them are as of that date.** "We now have a draft PR open on Microsoft's AutoGen" refers to microsoft/autogen#8009, which was opened by an independent contributor (@DresdenGman), not by this project. As of 2026-09-27 it is still an open, unmerged draft. The Inspect AI discussion ended in a docs listing (UKGovernmentBEIS/inspect_ai#4797), not a native integration. Current status: [`spec/ADOPTION.md`](../spec/ADOPTION.md).

---

## Follow-up 1: Jeffrey Ip (DeepEval)

**Subject:** Re: EvalPort + DeepEval — Portable Eval Datasets

Hi Jeffrey,

Just following up on my email from a few days ago about EvalPort.

Quick update: we now have a draft PR open on Microsoft's AutoGen implementing EvalPort import/export, and the Inspect AI (UK AISI) maintainer is actively discussing integration. The PyPI package (`pip install evalport-sdk`) and npm packages are live.

I filed an issue on the DeepEval repo (#2981) — would love to get your thoughts on whether native EvalPort import/export would be useful for DeepEval users.

Happy to jump on a quick call if easier. 15 minutes max.

Best,
Sahi

---

## Follow-up 2: Michael D'Amour (Promptfoo)

**Subject:** Re: EvalPort + Promptfoo — Portable Eval Datasets

Hi Michael,

Following up on my earlier email. We've made progress:

- Microsoft AutoGen has a draft PR implementing EvalPort import/export
- Inspect AI (UK AISI) maintainer is discussing integration
- Packages live on npm (`evalport-sdk`) and PyPI (`evalport-sdk`)
- 33 issues filed across the AI eval ecosystem

We already have a Promptfoo → EvalPort converter built. Would be great to get Promptfoo's perspective on whether native import/export is worth adding.

Issue filed: promptfoo/promptfoo#10235

Best,
Sahi

---

## Follow-up 3: EleutherAI

**Subject:** Re: EvalPort + lm-evaluation-harness — Portable Benchmark Results

Hi EleutherAI team,

Following up on my email about EvalPort. Quick progress update:

- Microsoft AutoGen has a draft PR implementing EvalPort import/export
- 33 issues filed across the AI eval ecosystem
- Packages published on npm and PyPI

For lm-evaluation-harness specifically, EvalPort could provide a standard ResultSet format so benchmark results from different models are directly comparable across tools. We have an issue open on your repo (#3962).

Would love to hear your thoughts.

Best,
Sahi
