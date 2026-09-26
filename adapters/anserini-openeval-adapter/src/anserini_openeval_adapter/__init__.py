"""Convert Anserini (https://github.com/castorini/anserini) TREC run/qrels
output to and from EvalPort (https://github.com/adhabnr-ux/evalport), the
open interchange format for portable LLM/IR evaluation datasets.

Why this parses text files rather than importing anserini
-----------------------------------------------------------
Anserini is a Java toolkit; its actual interchange point with the rest of
the IR ecosystem is the run file, not a Python object. Read directly from
``io/anserini/search/RunOutputWriter.java``, ``SearchCollection`` writes one
of two formats:

- Standard TREC run format: ``qid Q0 docid rank score runtag`` (one
  whitespace-separated line per retrieved document, written as
  ``"%s Q0 %s %d %f %s\\n"``).
- The ``msmarco`` variant: ``qid\\tdocid\\trank`` (tab-separated, no score
  or run tag column).

Paired with a standard TREC qrels file (``qid 0 docid relevance``, the same
format ``trec_eval`` and every other TREC-ecosystem tool consumes), that's
enough to build an EvalPort ``EvalSuite`` (one ``TestCase`` per query, its
retrieved ranking as ``retrieval_context``) and an EvalPort ``ResultSet``
(one ``Result`` per query, scored against the qrels).

This package has **zero import dependency on Anserini itself** — it only
reads the public run-file/qrels-file text formats Anserini already writes,
the same "parse the interchange format, don't import the framework" design
``autogen-openeval-adapter`` and ``pyserini-openeval-adapter`` both already
use in this repo, for the same reason: no JVM, no Anserini install, no
Python bindings required just to convert a run someone already produced.

Origin: raised as an Idea against Anserini itself
(castorini/anserini#3448, castorini/anserini#3449); Anserini's maintainer
confirmed the adapter belongs here, not in Anserini
("I'm happy for others to build on Anserini, but this should live in
EvalPort").

On IR metrics — honest scope
-----------------------------
Anserini itself does not compute effectiveness metrics (MAP, nDCG, recall)
from a run+qrels pair; that has always been ``trec_eval``'s job downstream,
and ``SearchCollection`` only ever writes the ranked list. This adapter
fills that gap with **its own small, pure-Python, dependency-free
implementations** of MAP, nDCG@k, Recall@k, and MRR (see ``metrics.py``-style
functions below) so ``to_openeval()`` can produce a real, scored
``ResultSet`` without requiring a JDK, a bundled ``trec_eval`` binary, or
``pytrec_eval``.

These are standard, documented formulas (linear-gain nDCG with a
``log2(rank + 1)`` discount; per-relevant-rank average precision; qrels-count
recall; reciprocal rank of the first relevant hit) — not a reimplementation
of trec_eval's C source, and this module makes no claim to reproduce
trec_eval's binary output bit-for-bit (trec_eval itself has shipped more
than one nDCG gain-function convention over the years). Every
``grader_result`` this module produces is tagged ``type: "custom"`` with
``params.handler = "anserini_openeval_adapter:<metric>"`` precisely so a
consumer that needs trec_eval-exact numbers knows to treat this as this
adapter's own scoring, not trec_eval's — matching the escape hatch the
EvalPort spec defines for exactly this situation. Anyone who wants
trec_eval-exact scores instead should compute them with
``pyserini.eval.trec_eval`` (or a JDK-based trec_eval install) and hand the
result to this repo's own ``pyserini-openeval-adapter``, which converts
trec_eval's *own* per-query output.
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openeval.types import OPENEVAL_VERSION
except ImportError:  # pragma: no cover - evalport-sdk always required at runtime,
    # but keep a sane fallback for static analysis / partial installs.
    OPENEVAL_VERSION = "1.0.0"

__all__ = [
    "parse_run_file",
    "parse_qrels_file",
    "parse_topics_file",
    "average_precision",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "to_openeval",
    "from_openeval",
    "__version__",
]
__version__ = "0.1.0"

DEFAULT_METRICS: Tuple[str, ...] = ("map", "ndcg_cut_10", "recall_100")
_METRIC_K = {
    # cutoff encoded in the metric's own name, trec_eval-style ("ndcg_cut_10"
    # -> k=10). "map" and "mrr" are computed over the full retrieved list.
}


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------


def parse_run_file(path: str) -> "Dict[str, List[Tuple[str, int, Optional[float]]]]":
    """Parse a TREC run file written by Anserini's ``RunOutputWriter``.

    Auto-detects the two formats it writes:

    - Standard TREC (6 whitespace-separated fields): ``qid Q0 docid rank
      score runtag``.
    - ``msmarco`` (3 tab-separated fields, no score/runtag): ``qid\\tdocid\\trank``.

    Returns ``{qid: [(docid, rank, score_or_None), ...]}``, each list sorted
    by rank ascending (rank 1 = top). Blank lines are skipped. Duplicate
    ``(qid, docid)`` pairs keep the first (best-ranked) occurrence, matching
    trec_eval's own tolerance of duplicate-safe run files.
    """
    runs: "Dict[str, List[Tuple[str, int, Optional[float]]]]" = {}
    seen: "Dict[str, set]" = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) >= 6:
                qid, _q0, docid, rank_s, score_s, *_rest = fields[:6]
                rank = int(rank_s)
                score: Optional[float] = float(score_s)
            elif len(fields) == 3:
                qid, docid, rank_s = fields
                rank = int(rank_s)
                score = None
            else:
                raise ValueError(
                    f"{path}:{lineno}: unrecognized run-file line "
                    f"(expected 6 TREC fields or 3 msmarco fields, got "
                    f"{len(fields)}): {raw_line!r}"
                )
            runs.setdefault(qid, [])
            dup = seen.setdefault(qid, set())
            if docid in dup:
                continue
            dup.add(docid)
            runs[qid].append((docid, rank, score))
    for qid in runs:
        runs[qid].sort(key=lambda t: t[1])
    return runs


def parse_qrels_file(path: str) -> "Dict[str, Dict[str, int]]":
    """Parse a standard TREC qrels file: ``qid iteration docid relevance``.

    The ``iteration`` column (conventionally ``0``) is read and discarded,
    matching trec_eval's own qrels format. Returns
    ``{qid: {docid: relevance}}`` with ``relevance`` as ``int``.
    """
    qrels: "Dict[str, Dict[str, int]]" = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{lineno}: expected 4 fields "
                    f"(qid iteration docid relevance), got {len(fields)}: "
                    f"{raw_line!r}"
                )
            qid, _iteration, docid, rel_s = fields
            qrels.setdefault(qid, {})[docid] = int(rel_s)
    return qrels


def parse_topics_file(path: str) -> "Dict[str, str]":
    """Parse a simple two-column topics file: ``qid<TAB>query text`` per line.

    This is deliberately the simplified tab-separated topics format used
    across the IR ecosystem (e.g. BEIR's ``queries.tsv``), not the SGML/XML
    TREC topics format Anserini's own ``-topics`` flag also accepts — this
    adapter only needs query *text* for ``TestCase.input``, not the full
    topic structure (title/description/narrative). Convert a native TREC
    topics file with ``bin/trec2json_topics.sh`` (or extract the ``<title>``
    field yourself) to this format first if that's what you have.
    """
    topics: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(
                    f"{path}:{lineno}: expected 'qid<TAB>query text', no tab found: "
                    f"{raw_line!r}"
                )
            qid, text = line.split("\t", 1)
            topics[qid.strip()] = text.strip()
    return topics


# ---------------------------------------------------------------------------
# Metrics (pure Python, no trec_eval/pytrec_eval dependency — see module
# docstring "On IR metrics — honest scope")
# ---------------------------------------------------------------------------


def average_precision(ranked_docids: Sequence[str], qrels_for_query: Dict[str, int]) -> float:
    """Average Precision: mean of precision@k at every rank a relevant
    document is retrieved. 0.0 if the query has no judged-relevant document
    (matches trec_eval's own convention for an unjudged/non-relevant topic).
    """
    relevant = {d for d, rel in qrels_for_query.items() if rel > 0}
    if not relevant:
        return 0.0
    hits = 0
    precisions: List[float] = []
    for i, docid in enumerate(ranked_docids, start=1):
        if docid in relevant:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / len(relevant) if precisions else 0.0


def _dcg(gains: Sequence[float]) -> float:
    # DCG@k = sum_{i=1}^{k} gain_i / log2(i + 1) — the modern (all-positions
    # logarithmically discounted) formulation; see module docstring for why
    # this, rather than trec_eval's own gain-function convention, is used.
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked_docids: Sequence[str], qrels_for_query: Dict[str, int], k: int) -> float:
    """nDCG@k using graded relevance straight from qrels as the gain (linear
    gain, not exponential), discounted by ``log2(rank + 1)``. 0.0 if the
    ideal DCG (i.e. every judged-relevant document) is 0.
    """
    top_k = list(ranked_docids)[:k]
    gains = [max(0, qrels_for_query.get(d, 0)) for d in top_k]
    dcg = _dcg(gains)
    ideal_gains = sorted((rel for rel in qrels_for_query.values() if rel > 0), reverse=True)[:k]
    idcg = _dcg(ideal_gains)
    return (dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(ranked_docids: Sequence[str], qrels_for_query: Dict[str, int], k: int) -> float:
    """Recall@k: fraction of all judged-relevant documents present in the
    top k retrieved. 0.0 if the query has no judged-relevant document.
    """
    relevant = {d for d, rel in qrels_for_query.items() if rel > 0}
    if not relevant:
        return 0.0
    top_k = set(list(ranked_docids)[:k])
    return len(relevant & top_k) / len(relevant)


def reciprocal_rank(ranked_docids: Sequence[str], qrels_for_query: Dict[str, int]) -> float:
    """1 / rank of the first relevant document in the full ranked list, or
    0.0 if none of the retrieved documents are judged relevant.
    """
    relevant = {d for d, rel in qrels_for_query.items() if rel > 0}
    if not relevant:
        return 0.0
    for i, docid in enumerate(ranked_docids, start=1):
        if docid in relevant:
            return 1.0 / i
    return 0.0


def _compute_metric(name: str, ranked_docids: Sequence[str], qrels_for_query: Dict[str, int]) -> float:
    if name == "map":
        return average_precision(ranked_docids, qrels_for_query)
    if name == "mrr":
        return reciprocal_rank(ranked_docids, qrels_for_query)
    if name.startswith("ndcg_cut_"):
        k = int(name[len("ndcg_cut_") :])
        return ndcg_at_k(ranked_docids, qrels_for_query, k)
    if name.startswith("recall_"):
        k = int(name[len("recall_") :])
        return recall_at_k(ranked_docids, qrels_for_query, k)
    raise ValueError(
        f"anserini_openeval_adapter: unknown metric {name!r}. Supported: "
        "'map', 'mrr', 'ndcg_cut_K', 'recall_K' (K an integer, e.g. "
        "'ndcg_cut_10', 'recall_100')."
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# to_openeval / from_openeval
# ---------------------------------------------------------------------------


def to_openeval(
    run_file: str,
    qrels_file: str,
    topics_file: Optional[str] = None,
    metrics: Sequence[str] = DEFAULT_METRICS,
    suite_id: str = "anserini_suite",
    run_id: Optional[str] = None,
    retrieval_context_depth: int = 100,
    pass_threshold: float = 0.0,
    version: str = OPENEVAL_VERSION,
) -> "Tuple[Dict[str, Any], Dict[str, Any]]":
    """Convert an Anserini TREC run file + qrels file (+ optional topics
    file) into an EvalPort ``(EvalSuite, ResultSet)`` pair, both as plain
    dicts conforming to the EvalPort schema.

    Args:
        run_file: Path to a run file written by Anserini's
            ``RunOutputWriter`` (standard TREC or ``msmarco`` format;
            auto-detected — see ``parse_run_file``).
        qrels_file: Path to a standard TREC qrels file.
        topics_file: Optional path to a simple ``qid<TAB>query text`` topics
            file (see ``parse_topics_file``). When omitted, each
            ``TestCase.input`` falls back to the query id itself.
        metrics: Metric names to score each query on. Supported: ``"map"``,
            ``"mrr"``, ``"ndcg_cut_K"``, ``"recall_K"`` for any integer K.
            Defaults to ``("map", "ndcg_cut_10", "recall_100")``.
        suite_id: EvalPort ``EvalSuite.id`` / ``ResultSet.suite_id``.
        run_id: EvalPort ``ResultSet.run_id``; a random one is generated if
            omitted.
        retrieval_context_depth: How many top-ranked docids to keep in each
            ``TestCase.retrieval_context`` (metrics are still computed over
            the query's *entire* run, not just this many).
        pass_threshold: A query "passes" a metric when its value is
            ``> pass_threshold`` (default ``0.0``). IR metrics have no
            universal "good" value — override this for your collection
            rather than relying on the default for anything but a basic
            non-zero check.
        version: EvalPort schema version to stamp onto both documents.

    Returns:
        ``(suite, result_set)`` — both plain dicts. Validate with
        ``openeval.validate.validate_suite(suite)`` /
        ``validate_result_set(result_set)``.

    Raises:
        ValueError: if the run file and qrels file share no query ids, or a
            requested metric name isn't recognized.
    """
    runs = parse_run_file(run_file)
    qrels = parse_qrels_file(qrels_file)
    topics = parse_topics_file(topics_file) if topics_file else {}

    qids = sorted(runs.keys())
    if not qids:
        raise ValueError(f"anserini_openeval_adapter: {run_file} has no queries.")
    if not any(qid in qrels for qid in qids):
        raise ValueError(
            "anserini_openeval_adapter: no query id in the run file has a "
            f"matching entry in the qrels file ({qrels_file}); double-check "
            "both files come from the same topic set."
        )

    graders = [
        {
            "id": f"gr_{m}",
            "type": "custom",
            "params": {"handler": f"anserini_openeval_adapter:{m}"},
            "description": f"Anserini/TREC retrieval metric '{m}', scored by anserini-openeval-adapter.",
        }
        for m in metrics
    ]
    grader_ids = [g["id"] for g in graders]

    test_cases: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    for qid in qids:
        ranked = [docid for docid, _rank, _score in runs[qid]]
        qrels_for_query = qrels.get(qid, {})

        tc: Dict[str, Any] = {
            "id": qid,
            "input": topics.get(qid, qid),
            "graders": list(grader_ids),
            "retrieval_context": ranked[:retrieval_context_depth],
            "metadata": {
                "anserini": {
                    "num_retrieved": len(ranked),
                    "num_judged": len(qrels_for_query),
                }
            },
        }
        test_cases.append(tc)

        grader_results: List[Dict[str, Any]] = []
        for m in metrics:
            score = _compute_metric(m, ranked, qrels_for_query)
            grader_results.append(
                {
                    "grader_id": f"gr_{m}",
                    "type": "custom",
                    "score": score,
                    "passed": score > pass_threshold,
                    "metadata": {"anserini": {"metric": m}},
                }
            )
        results.append(
            {
                "test_case_id": qid,
                "passed": all(gr["passed"] for gr in grader_results),
                "grader_results": grader_results,
                "metadata": {"anserini": {"num_relevant_judged": sum(1 for r in qrels_for_query.values() if r > 0)}},
            }
        )

    suite: Dict[str, Any] = {
        "version": version,
        "id": suite_id,
        "name": f"Anserini run: {suite_id}",
        "test_cases": test_cases,
        "graders": graders,
        "metadata": {"openeval": {"source": "anserini"}, "anserini": {"run_file": run_file, "qrels_file": qrels_file}},
    }

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    result_set: Dict[str, Any] = {
        "version": version,
        "suite_id": suite_id,
        "run_id": run_id or f"anserini_run_{uuid.uuid4().hex[:12]}",
        "started_at": _now_iso(),
        "results": results,
        "summary": {
            "total": total,
            "passed": passed_count,
            "failed": total - passed_count,
            "pass_rate": (passed_count / total) if total else 0.0,
        },
    }

    return suite, result_set


def from_openeval(suite: Dict[str, Any], runtag: str = "anserini_openeval_adapter") -> str:
    """Convert an EvalPort suite back into TREC standard run-file text.

    Reconstructs one run line per ``(test_case, retrieved doc)`` pair from
    each ``TestCase.retrieval_context`` (using ``TestCase.id`` as the query
    id, rank as the doc's position in that list, and a synthetic
    descending score of ``1 / rank`` since EvalPort's schema doesn't carry
    the original retrieval score — this is a documented, one-way-lossy
    round trip: docids and their relative order survive; the original
    scores and run tag do not).

    Returns the run file content as a single string (one line per row,
    ``qid Q0 docid rank score runtag``); write it to disk yourself if you
    need a file.

    Raises:
        ValueError: if a test case has no ``retrieval_context``.
    """
    lines: List[str] = []
    for tc in suite.get("test_cases", []):
        qid = tc.get("id")
        docids = tc.get("retrieval_context")
        if not docids:
            raise ValueError(
                f"anserini_openeval_adapter.from_openeval: test case {qid!r} has "
                "no retrieval_context to reconstruct a run from."
            )
        for rank, docid in enumerate(docids, start=1):
            score = 1.0 / rank
            lines.append(f"{qid} Q0 {docid} {rank} {score:.6f} {runtag}")
    return "\n".join(lines) + ("\n" if lines else "")
