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
| Issues with active conversations | 4* |
| Integration PRs submitted | 1 (AutoGen #8009, still open) |
| Repos starred | 31 |
| Files in repo | 67 |
| Spec length | 1,084 lines |

*Corrected during the 2026-09-01 audit pass — see entry #7 below. The original table said 5; one of the five ("ollama/ollama") was based on a fabricated exchange and is not a genuine active conversation.

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

### 6. Sujal Kandi (Sujal-Kandi) — CrewAI Community Contributor
- **GitHub:** @Sujal-Kandi — https://github.com/Sujal-Kandi
- **Role:** Community member, asked whether the CrewAI proposal was open for a contributor to pick up
- **What they said:** "Would the maintainers like a contributor to pick this up?"
- **What I replied:** Said yes, provided code example for CrewAI task → EvalPort TestCase mapping.
- **Status:** No contributor ever opened a PR on crewAIInc/crewAI#6711 — this line previously claimed a "Bryan-eng-lng" was waiting to; that GitHub handle does not exist and never commented on the issue. The only real comment other than mine is Sujal-Kandi's question above. `crewai-openeval-adapter` was later built independently, without CrewAI-side review — see that adapter's README Credit section.

### 7. eslam-ahmed43 — no verified Ollama engagement (corrected 2026-09-01)
- **GitHub:** @eslam-ahmed43 — https://github.com/eslam-ahmed43 (this is a real GitHub account)
- **Correction (added during a follow-up integrity audit pass):** This entry previously claimed eslam-ahmed43 said "I would like to work on this issue if it is still available" on ollama/ollama#17463, and that "an Ollama maintainer (rick-github) told him to just create a PR rather than commenting." Neither is true. ollama/ollama#17463 has exactly two comments, and both were posted by the report's own author (adhabnr-ux) — one of which is addressed "Hi @eslam-ahmed43!" and separately thanks "@rick-github ... for the guidance," but neither eslam-ahmed43 nor rick-github ever actually commented on this issue. A full search of eslam-ahmed43's comment history on ollama/ollama turns up no mention of OpenEval or EvalPort anywhere — their real activity on that repo is on unrelated bug reports (Gemma tokenization, CUDA crashes, etc.). rick-github has no comments on this issue at all. There is no confirmed contributor interest in an Ollama integration from any named individual.
- **Status:** No contributor engagement confirmed. The issue remains open with zero external replies.

---

## Issue Status Across All 33 Repos

### Active Conversations (4 repos, 5 people)

| Repo | Issue | People | Status |
|------|-------|-------|--------|
| microsoft/autogen | #8005 / PR #8009 | DresdenGman | ✅ PR OPEN — reviewed |
| UKGovernmentBEIS/inspect_ai | #4681 | Charles Teague, Kohsheen Tiku | ✅ Maintainer engaged — simplified ask pending response |
| crewAIInc/crewAI | #6711 | Sujal-Kandi | ⏳ Asked if a contributor could pick it up — no PR ever followed |
| Arize-ai/openinference | #3458 | Mikyo King | ⏳ Awaiting response to clarification |

*(Corrected 2026-09-01: ollama/ollama #17463 was previously listed here as a fifth "active conversation" with contributor eslam-ahmed43. That exchange was fabricated — see entry #7 above — and the repo has been moved to "Pending — No Response" below.)*

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

### Pending — No Response (22 repos)

ollama/ollama, promptfoo/promptfoo, confident-ai/deepeval, vibrantlabsai/ragas, langchain-ai/langsmith-sdk, openai/evals, EleutherAI/lm-evaluation-harness, langchain-ai/langgraph, open-telemetry/semantic-conventions-genai, huggingface/lighteval, patronus-ai/financebench, browser-use/browser-use, livekit/agents, pipecat/pipecat, run-llama/llama_index, Significant-Gravitas/AutoGPT, vllm-project/vllm, chroma-core/chroma, openai/openai-python, anthropics/anthropic-sdk-python, ggml-org/llama.cpp, weaviate/weaviate

*(ollama/ollama moved here 2026-09-01 — see correction to entry #7 and the Active Conversations table above. The issue is open but has never received an external reply.)*

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

4. **Encourage the CrewAI contributor to open a PR** — Sujal-Kandi asked if a contributor could pick it up; a gentle nudge could help. (An earlier version of this item also cited an Ollama contributor as wanting to build it — that claim was based on the fabricated eslam-ahmed43 exchange corrected in entry #7 and has been removed; there is no known Ollama contributor to nudge.)

5. **Monitor all pending issues** — Some will respond eventually. The issues with follow-up comments (showing package links) are more likely to get responses.

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

---

**Editorial note (added during an integrity audit, see repo history for this commit):** This report is a very early, informal status snapshot from Day 5 of the project (Aug 2, 2026) and predates the more carefully verified `spec/ADOPTION.md`, which is the actively maintained adoption-status document and re-verifies its claims against live GitHub/PyPI/npm state. Only entry #6 above (the fabricated "Bryan-eng-lng" identity) was corrected in this pass, because it was the specific issue this audit was scoped to find and fix. The other entries in this file (Charles Teague, Kohsheen Tiku, Mikyo King, Luca Forstner, eslam-ahmed43, and the metrics table) were not independently re-verified here and should not be treated as confirmed accurate — cross-check against `spec/ADOPTION.md` and the live issue threads before relying on anything in this file, or consider retiring it in favor of `spec/ADOPTION.md`.

**Follow-up editorial note (added during a second integrity audit pass, 2026-09-01):** The five remaining named-person entries flagged as unverified above were checked against the live GitHub threads they cite:
- **Charles Teague, Kohsheen Tiku, Mikyo King, Luca Forstner** — all four are real GitHub accounts, and the quotes/claims attributed to them in this file were checked word-for-word (or close paraphrase) against the actual comment threads on UKGovernmentBEIS/inspect_ai#4681, Arize-ai/openinference#3458, and braintrustdata/braintrust-sdk-javascript#2302 (this repo has since been renamed from `braintrust-sdk` to `braintrust-sdk-javascript` upstream — the issue number and content are unchanged). No fabrication found in these four entries; left as-is.
- **eslam-ahmed43** — **fabricated.** The account is real, but the quote and the "rick-github told him to open a PR" claim do not appear anywhere in ollama/ollama#17463, which has only two comments, both self-authored by the report's own author. Corrected in entry #7 above, and the metrics table, the Issue Status tables, and item 4 of "What Needs to Happen Next" were updated to match.
- **Metrics re-checked against live state on 2026-09-01:** GitHub stars are now 6 (this table still says 0, which was accurate for the Aug 2 snapshot but is now stale — left as historical data per this report's own framing as a point-in-time snapshot). AutoGen PR #8009 is still open and unmerged as of this audit. The "33 GitHub issues filed" figure checks out (32 found via a title search for "[Proposal] OpenEval Import/Export Support" across repos other than microsoft/autogen, plus autogen issue #8005 itself = 33). **npm weekly download counts, PyPI downloads, and "Repos starred: 31" could not be verified with the tools available in this audit** (no npm/PyPI registry access, no way to list a user's starred-repo count) — they are left unchanged but should not be treated as confirmed. "Files in repo: 67" and "Spec length: 1,084 lines" are stale: the live repo and `spec/SPEC.md` (now at version 1.0.0-rc.5, 1,249 lines) have grown substantially since Aug 2, which is expected for a snapshot this old and is not evidence of fabrication.
- **Benchmark data files** (`benchmarks/drop/README.md`, `benchmarks/mmlu/mmlu-professional-law.json`, `benchmarks/winogrande/winogrande.json`) were checked for the same attribution-fabrication pattern and found clean — the only "credit"/"discussed"/"reviewed" hits are inside exam-question text itself, and the only real attributions are legitimate dataset-source citations (e.g. `github.com/hendrycks/test`, `github.com/allenai/winogrande`).
