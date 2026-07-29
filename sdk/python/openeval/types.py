from __future__ import annotations
from typing import Any, Literal, Union, Optional, List, Dict
from dataclasses import dataclass, field

OPENEVAL_VERSION = "1.0.0"

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
    summary: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    valid: bool
    errors: List[Dict[str, str]] = field(default_factory=list)
