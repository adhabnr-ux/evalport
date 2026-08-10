# Benchmark Licenses & Attribution

Every benchmark converted into this hub is included **only after its license
was independently verified from its primary source** — the dataset's own
repository / dataset-card "Licensing Information" section, or the paper's
stated release terms. The license strings below are not taken from any
third-party aggregator or assumed from a benchmark's popularity; each was
confirmed by fetching the source directly during conversion. Where a source
had ambiguous or missing licensing information, it was **excluded** rather
than included on a best guess — see "Excluded candidates" at the bottom.

If you believe any entry below is inaccurate, please open an issue — this
table is treated as a hard gate on what's allowed to live in `benchmarks/`,
not a convenience note.

## Included benchmarks

| Benchmark | License | Source | Paper |
|---|---|---|---|
| GSM8K | MIT | [openai/grade-school-math](https://github.com/openai/grade-school-math) | [arXiv:2110.14168](https://arxiv.org/abs/2110.14168) |
| ARC (Easy + Challenge) | CC-BY-SA-4.0 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | [arXiv:1803.05457](https://arxiv.org/abs/1803.05457) |
| BoolQ | CC-BY-SA-3.0 | [google-research-datasets/boolean-questions](https://github.com/google-research-datasets/boolean-questions) | [arXiv:1905.10044](https://arxiv.org/abs/1905.10044) |
| HellaSwag | MIT | [rowanz/hellaswag](https://github.com/rowanz/hellaswag) | [arXiv:1905.07830](https://arxiv.org/abs/1905.07830) |
| WinoGrande | CC-BY | [allenai/winogrande](https://github.com/allenai/winogrande) | [arXiv:1907.10641](https://arxiv.org/abs/1907.10641) |
| CommonsenseQA | MIT | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | [arXiv:1811.00937](https://arxiv.org/abs/1811.00937) |
| PIQA | AFL-3.0 | [yonatanbisk.com/piqa](https://yonatanbisk.com/piqa/) | [arXiv:1911.11641](https://arxiv.org/abs/1911.11641) |
| TruthfulQA | Apache-2.0 | [truthfulqa/truthful_qa](https://huggingface.co/datasets/truthfulqa/truthful_qa) | [arXiv:2109.07958](https://arxiv.org/abs/2109.07958) |
| MMLU | MIT | [hendrycks/test](https://github.com/hendrycks/test) | [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) |
| HumanEval | MIT | [openai/human-eval](https://github.com/openai/human-eval) | [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) |
| MBPP | CC-BY-4.0 | [google-research/mbpp](https://github.com/google-research/google-research/tree/master/mbpp) | [arXiv:2108.07732](https://arxiv.org/abs/2108.07732) |
| SQuAD 2.0 | CC-BY-SA-4.0 | [rajpurkar.github.io/SQuAD-explorer](https://rajpurkar.github.io/SQuAD-explorer/) | [arXiv:1806.03822](https://arxiv.org/abs/1806.03822) |
| DROP | CC-BY-SA-4.0 | [ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop) | [arXiv:1903.00161](https://arxiv.org/abs/1903.00161) |
| BIG-Bench Hard (BBH) | MIT | [suzgunmirac/BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | [arXiv:2210.09261](https://arxiv.org/abs/2210.09261) |

All 14 licenses above (MIT, Apache-2.0, CC-BY, CC-BY-4.0, CC-BY-SA-3.0,
CC-BY-SA-4.0, AFL-3.0) permit redistribution of the data with attribution,
which is why each is included here. This repository redistributes only the
fields needed to reconstruct an EvalPort test case (question/prompt,
reference answer, and — for CC-BY-SA sources — the passage text the question
was originally written against); it does not redistribute any other content
from the source repositories.

## Notes on specific entries

- **ARC / MMLU / HellaSwag / WinoGrande / SQuAD 2.0 / DROP** are all licensed
  CC-BY-SA or CC-BY, which requires **share-alike / attribution** on
  redistribution. This file *is* that attribution; if you redistribute these
  suites further, carry this file (or equivalent attribution) with them.
- **PIQA** required extra care during verification: there are two unrelated
  projects that both go by "PIQA" on GitHub — Yonatan Bisk's Physical
  Interaction QA (the commonsense-reasoning benchmark converted here, hosted
  at [yonatanbisk.com/piqa](https://yonatanbisk.com/piqa/), AFL-3.0) and a
  separate "Phrase-Indexed Question Answering" project under a similarly
  named GitHub repo. The license above was confirmed against Bisk's actual
  physical-commonsense PIQA, not the unrelated phrase-indexed QA project.
- **TruthfulQA** is hosted at the namespaced HF repo id
  `truthfulqa/truthful_qa` (the older un-namespaced `truthful_qa` repo id no
  longer resolves) — Apache-2.0 per that repo's dataset card.
- **HumanEval** and **MBPP** ship `code`-type graders (a pass/fail unit-test
  harness per test case) rather than `exact_match`/`contains` — see each
  benchmark's own README for why, and note that running these suites
  requires a runner with sandboxed Python execution; the EvalPort spec's
  `unsupported_grader_type` skip mechanism lets runners without one skip
  these cleanly instead of failing.

## Excluded candidates

- **LAMBADA** was considered (it appears on many "standard eval suite"
  lists) but excluded from this hub. Its data is derived from a filtered
  subset of the BookCorpus, whose own redistribution rights are unclear and
  contested — several BookCorpus mirrors have been taken down over
  copyright-provenance concerns, and no LAMBADA source we checked offered an
  unambiguous redistribution license independent of that underlying
  provenance question. Given the hard rule that only benchmarks whose
  license *permits redistribution of the data* are included, LAMBADA was
  left out rather than included on an assumption.

## How this was verified

Every license above was checked directly against a primary source during
conversion — either a HuggingFace dataset card's "Licensing Information"
section or the GitHub repository's own `LICENSE` file / README licensing
statement — not inferred from a benchmark's name, popularity, or how other
eval frameworks classify it. Where the initial parenthetical hint used to
scope this work turned out to reference the wrong project entirely (the PIQA
naming collision above), the license shown here reflects the corrected,
verified source, not the original hint.
