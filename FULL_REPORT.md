# EvalPort RFC — Full Status Report
**August 2, 2026 (Day 5 since launch)**

---

## Executive Summary

EvalPort is an open standard for portable LLM evaluation datasets, launched July 28, 2026. In 5 days, it has gone from zero to: a complete specification, published packages on npm and PyPI, 33 GitHub issues filed across the AI ecosystem, 1 integration PR submitted to Microsoft's AutoGen framework, and active conversations with 5 framework maintainers/contributors.

---

## Metrics

| Metric | Value |
|--------|-------|
| GitHub stars | 0 |
| GitHub forks | 0 |
| npm weekly downloads (evalport-sdk) | 139 |
| npm weekly downloads (evalport-cli) | ~50 (est) |
| PyPI downloads | N/A (new package) |
| GitHub issues filed | 33 |
| Issues with active conversations | 5 |
| Integration PRs submitted | 1 (AutoGen #8009, still open) |
| Repos starred | 31 |
| Files in repo | 67 |
| Spec length | 1,084 lines |

---

## Collaborators — Who They Are

### 1. Dresden (DresdenGman) — AutoGen Contributor
- **GitHub:** @DresdenGman — https://github.com/DresdenGman
- **Followers:** 3 | **Public repos:** 47 | **Joined:** May 2025
- **Role:** Independent contributor, opened the FIRST EvalPort integration PR
- **What they did:** Submitted PR #8009 to microsoft/autogen implementing `to_openeval()` / `from_openeval()` in `autogenstudio/eval/openeval.py` with tests and `evalport-sdk` as a dependency
- **Status:** PR is OPEN, awaiting AutoGen maintainer review. I reviewed the PR as spec author with 3 minor suggestions (non-blocking).
- **Significance:** This is the first real-world implementation of the EvalPort spec in an external framework.

### 2. Charles Teague (dragonstyle) — Inspect AI Maintainer
- **GitHub:** @dragonstyle — https://github.com/dragonstyle
- **Location:** Boston, MA | **Followers:** 146 | **Public repos:** 68 | **Joined:** May 2010
- **Role:** Maintainer of Inspect AI (UK Government AI Safety Institute)
- **What they said:** "We're definitely open to an extension" — asked about the benefit vs just installing EvalPort directly, and about name confusion with open-eval.com
- **What I replied:** Agreed the extension package was unnecessary (the SDK already has a converter), simplified ask to just doc cross-references. Proposed name alternatives (EvalPort, EvalSpec, PortableEval).
- **Status:** Waiting on his response to the simplified proposal. Last comment was ~15 hours ago.
- **Significance:** A UK government AI institute endorsing the spec would be the strongest credibility signal. Charles is an experienced maintainer (146 followers, 14 years on GitHub).

### 3. Kohsheen Tiku (kohsheen1234) — Inspect AI Community Contributor
- **GitHub:** @kohsheen1234 — https://github.com/kohsheen1234
- **Followers:** 1 | **Public repos:** 88 | **Joined:** Sep 2016
- **Role:** Community contributor, wants to build the Inspect AI integration
- **What they said:** "Would love to work on this! :)"
- **Status:** Linked to the maintainer conversation. If Charles greenlights, Kohsheen would be the one building it.

### 4. Mikyo King (mikeldking) — Arize/OpenInference Maintainer
- **GitHub:** @mikeldking — https://github.com/mikeldking
- **Bio:** "Head of OSS @Arize-ai" | **Location:** San Francisco | **Followers:** 137 | **Public repos:** 56 | **Joined:** Oct 2013
- **Role:** Head of Open Source at Arize AI, maintainer of OpenInference
- **What they said:** "I'm not quite following what this would mean. Can you elaborate?"
- **What I replied:** Detailed explanation of how EvalPort (eval data) complements OpenInference (execution traces), with a comparison table and integration examples.
- **Status:** Awaiting response. Last interaction was ~3 days ago.
- **Significance:** Arize is a major AI observability company. Mikyo is their Head of OSS — if Arize aligns EvalPort with OpenInference, it signals industry adoption.

### 5. Luca Forstner (lforst) — Braintrust SDK Maintainer
- **GitHub:** @lforst — https://github.com/lforst
- **Bio:** "now: @braintrustdata, prev: @supabase, @getsentry" | **Location:** Vienna, Austria | **Followers:** 124 | **Public repos:** 36 | **Joined:** Jul 2014
- **Role:** Engineer at Braintrust, maintainer of braintrust-sdk
- **What they said:** "Interesting way to sell this 😂 We have no intention of vendor locking folks, though right now this is not a big priority."
- **Status:** Issue closed as "not planned." Door left open for future.
- **Significance:** Braintrust is a well-funded eval platform. Luca is experienced (prev at Supabase and Sentry). Soft no for now, but not a rejection of the spec.

### 6. SUJAL KANDI (Bryan-eng-lng) — CrewAI Community Contributor
- **GitHub:** @Bryan-eng-lng — https://github.com/Bryan-eng-lng
- **Bio:** "AI Engineer | Agentic AI & LLM Systems | LangGraph · LangChain · RAG | Building autonomous agents that think, research, and self-correct" | **Followers:** 2 | **Public repos:** 20 | **Joined:** Jul 2025
- **Role:** New contributor, interested in agent eval
- **What they said:** "Would the maintainers like a contributor to pick this up?"
- **What I replied:** Said yes, provided code example for CrewAI task → EvalPort TestCase mapping.
- **Status:** Waiting for Bryan to open a PR. No action yet.

### 7. eslam-ahmed43 — Ollama Community Contributor
- **GitHub:** @eslam-ahmed43 — https://github.com/eslam-ahmed43
- **Followers:** 2 | **Public repos:** 10 | **Joined:** Feb 2025
- **Role:** Community contributor
- **What they said:** "I would like to work on this issue if it is still available."
- **What I replied:** Provided code example and encouraged PR.
- **Status:** Waiting for PR. An Ollama maintainer (rick-github) told him to just create a PR rather than commenting.

---

## Issue Status Across All 33 Repos

### Active Conversations (5 repos, 7 people)

| Repo | Issue | People | Status |
|------|-------|-------|--------|
| microsoft/autogen | #8005 / PR #8009 | DresdenGman | ✅ PR OPEN — reviewed |
| UKGovernmentBEIS/inspect_ai | #4681 | Charles Teague, Kohsheen Tiku | ✅ Maintainer engaged — simplified ask pending response |
| crewAIInc/crewAI | #6711 | Bryan-eng-lng | ✅ Contributor ready — awaiting PR |
| Arize-ai/openinference | #3458 | Mikyo King | ⏳ Awaiting response to clarification |
| ollama/ollama | #17463 | eslam-ahmed43 | ✅ Contributor ready — awaiting PR |

### Closed Gracefully (4 repos)

| Repo | Issue | Maintainer | Outcome |
|------|-------|-----------|---------|
| braintrustdata/braintrust-sdk | #2302 | Luca Forstner | Closed as "not planned" — door open |
| deepset-ai/haystack | #12193 | Julian Risch | Closed — "sounds interesting, no task yet" |
| mlflow/mlflow | #24719 | Tomu Hirata | Auto-closed by triage |
| wandb/wandb | #12329 | Dmitry Duev | Auto-closed by triage |

### Auto-Closed by Bots (3 repos)

| Repo | Issue | Outcome |
|------|-------|---------|
| OpenHands/OpenHands | #16189 | Duplicate-check bot |
| dotnet/runtime | #131570 | Wrong repo — closed by us |
| modelcontextprotocol | #3167 | Comment blocked by repo restrictions |

### Pending — No Response (21 repos)

promptfoo/promptfoo, confident-ai/deepeval, vibrantlabsai/ragas, langchain-ai/langsmith-sdk, openai/evals, EleutherAI/lm-evaluation-harness, langchain-ai/langgraph, open-telemetry/semantic-conventions-genai, huggingface/lighteval, patronus-ai/financebench, browser-use/browser-use, livekit/agents, pipecat/pipecat, run-llama/llama_index, Significant-Gravitas/AutoGPT, vllm-project/vllm, chroma-core/chroma, openai/openai-python, anthropics/anthropic-sdk-python, ggml-org/llama.cpp, weaviate/weaviate

---

## What's Published

| Package | Registry | URL | Downloads (last week) |
|---------|----------|-----|---------------------|
| evalport-sdk | npm | https://www.npmjs.com/package/evalport-sdk | 139 |
| evalport-cli | npm | https://www.npmjs.com/package/evalport-cli | ~50 |
| evalport-sdk | PyPI | https://pypi.org/project/evalport-sdk/1.0.0/ | New |

---

## What Needs to Happen Next (Priority Order)

1. **Get AutoGen PR #8009 merged** — This is THE milestone. "PR open" ≠ "PR merged." Once it merges, the resume line is real and defensible. Must stay responsive to any maintainer feedback.

2. **Get Charles Teague's response on Inspect AI** — A government AI institute listing EvalPort in their docs is a credibility multiplier. Follow up in 2-3 days if no response.

3. **Send founder email follow-ups** — 4 days have passed since initial emails to DeepEval, Promptfoo, and EleutherAI. Follow-up emails are drafted at `docs/founder-followups.md`.

4. **Encourage CrewAI and Ollama contributors to open PRs** — Both said they want to build it. A gentle nudge could help.

5. **Monitor all 21 pending issues** — Some will respond eventually. The issues with follow-up comments (showing package links) are more likely to get responses.

---

## Assessment

**Strengths:**
- First integration PR submitted to a Microsoft framework within 48 hours of launch
- Active conversation with a UK government AI institute maintainer
- 139 npm downloads in the first week (organic, from GitHub issues)
- Complete spec + SDKs + CLI + converters + documentation shipped

**Weaknesses:**
- 0 GitHub stars (social media reach was limited — Reddit blocked by karma)
- Only 1 PR (AutoGen) — need 2-3 more for real momentum
- No maintainer has explicitly endorsed or merged yet
- Name confusion risk with open-eval.com (Charles Teague raised this)

**What would make this a "success":**
- AutoGen PR merges → "Adopted by Microsoft AutoGen"
- Inspect AI adds doc cross-reference → "Supported by UK AISI"
- 2 more PRs from CrewAI/Ollama → "3 frameworks integrating"
- 100+ GitHub stars → visible community interest
