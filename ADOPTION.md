# OpenEval — Adoption Strategy

## Executive Summary

OpenEval adoption follows a three-phase strategy: **Build credibility** (reference implementation + early adopters), **Create gravity** (framework integrations + community), and **Standardize** (standards body submission). The goal is to reach the point where an eval framework that doesn't support OpenEval import/export is considered broken.

---

## Phase 1: Build Credibility (Months 1-3)

### 1.1 Launch

- Publish the spec, JSON Schemas, TypeScript SDK, Python SDK, and CLI to GitHub
- Publish `@openeval/sdk` to npm and `openeval` to PyPI
- Write a launch blog post: "Why LLM Evaluation Needs a Standard Format"
- Post on Hacker News, r/LocalLLaMA, r/MachineLearning, AI Twitter
- Create a simple landing page at openeval.org

### 1.2 First 20 GitHub Repositories to Target

| # | Repository | Why OpenEval Benefits Them | Maintainer/Champion | Priority |
|---|-----------|---------------------------|---------------------|----------|
| 1 | **promptfoo/promptfoo** | Import/export standard format; users can bring eval suites from other tools | Michael D'Amour (@tychedjs) | High |
| 2 | **confident-ai/deepeval** | Portability for DeepEval datasets; users can export to other runners | Jeffrey Ip (@jeffreyip) | High |
| 3 | **UKGovernmentBEIS/inspect_ai** | UK AISI's eval framework; standard format enables benchmark sharing | AISI team | High |
| 4 | **langchain-ai/langsmith-sdk** | LangSmith dataset import/export; cross-tool eval compatibility | LangChain team | High |
| 5 | **braintrustdata/braintrust-sdk** | Braintrust already has a generic eval format; OpenEval alignment reduces friction | Anand Kannan | Medium |
| 6 | **explodinggradients/ragas** | RAG eval standardization; users can use RAG-specific graders across tools | Shahul ES | Medium |
| 7 | **openai/evals** | OpenAI Evals format is widely used; OpenEval provides an upgrade path with provider-agnosticism | OpenAI evals team | Medium |
| 8 | **mlflow/mlflow** | MLflow eval tracking; OpenEval as a portable eval format for the MLOps ecosystem | MLflow maintainers | Medium |
| 9 | **Arize-ai/openinference** | Arize's OpenInference traces + OpenEval results = complete observability | Arize team | Medium |
| 10 | **microsoft/evaluator** | Microsoft's eval framework; enterprise users need portability | Microsoft AI team | Low |
| 11 | **patronus-ai/financebench** | Financial eval benchmarks need a portable format to be reusable | Patronus AI | Medium |
| 12 | **HazyResearch/llm-eval-harness** | Stanford eval harness; academic benchmarks need standard format | Hazy Research | Low |
| 13 | **lighteval** (HuggingFace) | HF's eval framework; OpenEval alignment enables dataset sharing on HF Hub | HuggingFace team | Medium |
| 14 | **EleutherAI/lm-evaluation-harness** | Most-used open LLM benchmark harness; portable results = comparable benchmarks | EleutherAI | High |
| 15 | **nlmatics/llm-eval** | Community eval tool; standard format lowers barrier to entry | Community | Low |
| 16 | **microsoft/autogen** | Agent eval; OpenEval's agent profile supports tool-call verification | AutoGen team | Low |
| 17 | **langchain-ai/langgraph** | LangGraph agent eval; OpenEval suite format for graph-based agents | LangChain | Low |
| 18 | **crewaiinc/crewai** | CrewAI agent eval; expected_tools field for multi-agent verification | CrewAI team | Low |
| 19 | **open-telemetry/semantic-conventions-genai** | Complementary: OTel for traces, OpenEval for eval data; cross-reference | OTel GenAI WG | Medium |
| 20 | **modelcontextprotocol** | MCP defines tools; OpenEval evaluates agents that use them; reference MCP tools in expected_tools | MCP team | Low |

### 1.3 Outreach Strategy

For each repository:
1. **File an issue** proposing OpenEval import/export support, linking to the spec and SDK
2. **Offer a PR** implementing the conversion (using the reference SDK)
3. **Engage in discussions** about format design, incorporating feedback into the spec

**Tone:** Not "adopt my standard" but "I built a converter that lets your users import/export eval datasets — would this be useful?"

### 1.4 Conference Venues

- **AI Engineer Summit** (San Francisco, Oct 2026) — primary venue; eval is a core topic
- **OSPO for AI** (Linux Foundation events) — standards-track presentation
- **Ray Summit** — MLflow/eval focus
- **MLOps World** — eval and observability track
- **PyData / SciPy** — Python data tooling audience
- **Local LLM meetups** — grassroots adoption

### 1.5 Mailing Lists and Forums

- LFAI & Data mailing list
- OpenTelemetry GenAI WG mailing list
- HuggingFace forums (eval dataset sharing)
- LangChain Discord (#evals channel)
- r/LocalLLaMA, r/MachineLearning
- Hacker News (launch post)

---

## Phase 2: Create Gravity (Months 4-12)

### 2.1 Framework Integrations

Target: At least 3 major frameworks ship native OpenEval import/export.

**Integration model:** Each framework adds two functions:
- `import_openeval_suite(path) -> FrameworkTestSuite`
- `export_openeval_suite(suite) -> OpenEvalSuite`

Plus optionally:
- `export_openeval_results(results) -> ResultSet`

### 2.2 Benchmark Registry

Create a public registry of OpenEval-format benchmark datasets at openeval.org/benchmarks. Each benchmark:
- Has a stable ID and version
- Is published as an OpenEval suite (JSON)
- Includes reference results from 2+ models
- Is citable in papers

This creates the "ImageNet for LLM eval" dynamic — a shared benchmark that works across tools.

### 2.3 Blog Strategy

1. **Launch post:** "OpenEval: A Standard Format for LLM Evaluation"
2. **Technical deep dive:** "How OpenEval's Grader Type System Works"
3. **Migration guide:** "Converting DeepEval Suites to OpenEval (and Back)"
4. **Case study:** "Running the Same Eval Suite in 5 Frameworks with OpenEval"
5. **Comparison:** "OpenEval vs. Every Eval Ever vs. OpenAI Evals: What's Different"
6. **Community post:** "How to Publish a Benchmark in OpenEval Format"

### 2.4 Documentation and Tutorials

- Getting started guide (5 minutes)
- Framework-specific integration guides (DeepEval, Promptfoo, Inspect, LangSmith)
- Grader type reference with examples
- Video tutorial: "From zero to portable evals in 10 minutes"
- Interactive playground at openeval.org/playground

### 2.5 GitHub Strategy

- Star the repo, pin it
- Create GitHub Discussions for spec feedback
- Use GitHub Projects for roadmap transparency
- Tag issues with `good-first-issue` for community contributions
- Create a `CONTRIBUTING.md` with clear guidelines

---

## Phase 3: Standardize (Months 13-24)

### 3.1 Linux Foundation AI & Data

Submit OpenEval to LFAI & Data as an incubating project. This provides:
- Governance structure
- Neutral IP ownership
- Industry credibility
- Ecosystem connections

**Precedent:** MCP was donated to Linux Foundation by Anthropic (Dec 2025). A2A was donated by Google. OpenEval follows the same path.

### 3.2 Working Group

Form an OpenEval Working Group with representatives from:
- At least 3 eval framework maintainers
- At least 1 standards body member (OTel GenAI WG, W3C, IETF)
- At least 1 enterprise user
- At least 1 academic researcher

### 3.3 Specification Governance

- Move spec to a dedicated repo (`openeval/spec`)
- Use RFC process for changes (proposal → discussion → acceptance)
- Semver for all schema changes
- Regular spec meetings (monthly)

### 3.4 IETF / W3C Track

Once adoption is proven (3+ frameworks, 1000+ datasets), submit as:
- IETF Internet-Draft (for the data format)
- W3C Community Group (for broader web ecosystem alignment)

---

## Likely Champions

| Person/Org | Why They'd Champion OpenEval |
|-----------|----------------------------|
| **Jeffrey Ip (DeepEval)** | DeepEval users frequently ask for export; OpenEval unblocks them |
| **Michael D'Amour (Promptfoo)** | Promptfoo already supports many formats; OpenEval is a natural addition |
| **UK AISI (Inspect AI)** | Government eval standards align with OpenEval's reproducibility goals |
| **LangChain team** | LangSmith datasets are already JSON; OpenEval makes them portable |
| **EleutherAI** | lm-eval-harness results need a standard format for cross-model comparison |
| **Arize** | OpenInference + OpenEval = complete observability stack |
| **HuggingFace** | HF Hub could host OpenEval benchmark datasets natively |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Frameworks refuse to adopt (NIH syndrome) | Lead with converters, not governance. Make it easier to add OpenEval than to build their own format. |
| OpenAI/Anthropic push their own format | Position OpenEval as provider-agnostic. Their formats can map to OpenEval via converters. |
| Spec is too complex | Keep v1 minimal. Only add grader types that 2+ frameworks implement. |
| Spec is too simple (doesn't cover edge cases) | Extension mechanism + custom grader type handle edge cases without bloating the core spec. |
| No one adopts | Target the pain point directly: "you can't share eval datasets" is a complaint every practitioner has. |

---

## Success Metrics

| Metric | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|---------------|---------------|---------------|
| GitHub stars | 500 | 5,000 | 10,000 |
| npm weekly downloads | 100 | 1,000 | 5,000 |
| PyPI weekly downloads | 50 | 500 | 2,000 |
| Frameworks with native support | 1 | 3 | 7+ |
| Published benchmark datasets | 5 | 50 | 200+ |
| Citations in papers | 0 | 5 | 20+ |

---

## Timeline

| Month | Milestone |
|-------|----------|
| 1 | Spec published, SDKs on npm/PyPI, launch blog post |
| 2 | First framework integration (Promptfoo or DeepEval) |
| 3 | 3 framework integrations, first benchmark dataset published |
| 6 | 5+ framework integrations, benchmark registry live |
| 9 | Conference presentations, working group formed |
| 12 | LFAI submission, 5000+ stars |
| 18 | LFAI incubation, governance transition |
| 24 | IETF/W3C submission, recognized standard |