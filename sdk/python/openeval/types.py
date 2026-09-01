from __future__ import annotations
from typing import Any, Literal, Union, Optional, List, Dict
from dataclasses import dataclass, field

# Tracks spec/SPEC.md's own **Version** header. Bumped to 1.0.0-rc.5 alongside the
# spec (see spec/SPEC.md Change Log). NOTE: this constant had drifted once before --
# it was still hardcoded to "1.0.0-rc.1" after spec/SPEC.md itself moved to
# 1.0.0-rc.2, meaning every document this SDK generated was silently stamping a
# stale spec version. Caught and fixed while implementing the 1.0.0-rc.3 changes
# (Discussions #9, #10, #11); test_convert.py now imports this constant instead of
# hardcoding a version literal, specifically so a bump like this one (Discussion
# #22, repetition/attempt tracking) can't drift the same way undetected.
OPENEVAL_VERSION = "1.0.0-rc.5"

GraderType = Literal["exact_match","contains","regex","semantic_similarity","llm_judge","json_schema","json_path","code","human","model graded","custom"]

@dataclass
class Grader:
    id: str
    type: GraderType
    params: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    description: Optional[str] = None

@dataclass
class TestCase:
    id: str
    input: Union[str, List[str]]
    graders: List[Union[str, Grader]]
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    timeout_ms: Optional[int] = None
    weight: float = 1.0

@dataclass
class EvalSuite:
    version: str
    id: str
    test_cases: List[TestCase]
    name: Optional[str] = None
    description: Optional[str] = None
    graders: List[Grader] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class GraderResult:
    grader_id: str
    type: str
    score: Optional[float]
    passed: bool
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Result:
    test_case_id: str
    passed: bool
    grader_results: List[GraderResult]
    actual_output: Optional[str] = None
    duration_ms: Optional[int] = None
    completed_at: Optional[str] = None
    # 1-indexed repetition number for this test_case_id within this run_id;
    # ascending = observation order. Absent means single-attempt (no change for
    # existing producers). Pairs with test_case_id + run_id as the join key for
    # repeated trials -- see Discussion #22 / spec/SPEC.md Extension Mechanism ->
    # Repetition & Attempt Tracking.
    attempt: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResultSet:
    version: str
    suite_id: str
    run_id: str
    started_at: str
    results: List[Result]
    suite_version: Optional[str] = None
    completed_at: Optional[str] = None
    provider: Optional[Dict[str, Any]] = None
    runner: Optional[Dict[str, str]] = None
    # Trial isolation mode ("fresh" | "shared" | any other open string) for
    # repeated attempts represented in `results` -- declared once per ResultSet,
    # not per Result, per Discussion #22 / issue #20: a ResultSet is one
    # collection of evidence and should make one isolation claim; a producer
    # that genuinely mixes isolation modes should emit two ResultSets instead.
    isolation: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    valid: bool
    errors: List[Dict[str, str]] = field(default_factory=list)
