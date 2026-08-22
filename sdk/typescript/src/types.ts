// Tracks spec/SPEC.md's own **Version** header. Bumped to 1.0.0-rc.4 alongside the
// spec (see spec/SPEC.md Change Log). NOTE: this constant had drifted once before --
// it was still hardcoded to "1.0.0-rc.1" after spec/SPEC.md itself moved to
// 1.0.0-rc.2, meaning every document this SDK generated was silently stamping a
// stale spec version. Caught and fixed while implementing the 1.0.0-rc.3 changes
// (Discussions #9, #10, #11); src/convert.ts imports this constant rather than
// hardcoding a version literal, and tests/convert.test.ts (added alongside this
// 1.0.0-rc.4 bump, mirroring the equivalent guard sdk/python/tests/test_convert.py
// already had) now asserts every document convert.ts produces stamps this exact
// value, so this kind of drift can't happen silently on the TypeScript side either.
export const OPENEVAL_VERSION = "1.0.0-rc.4";

export type GraderType =
  | "exact_match"
  | "contains"
  | "regex"
  | "semantic_similarity"
  | "llm_judge"
  | "json_schema"
  | "json_path"
  | "code"
  | "human"
  | "model graded"
  | "custom";

export interface ProviderConfig {
  model?: string;
  api_base?: string;
  api_key_env?: string;
  temperature?: number;
  max_tokens?: number;
  extra?: Record<string, unknown>;
}

export interface SuiteConfig {
  provider?: ProviderConfig;
  defaults?: {
    timeout_ms?: number;
    weight?: number;
  };
  parallel?: number;
  retry?: {
    max_attempts?: number;
    backoff_ms?: number;
  };
}

export interface GraderParams {
  [key: string]: unknown;
}

export interface Grader {
  id: string;
  type: GraderType;
  params?: GraderParams;
  weight?: number;
  description?: string;
}

export interface TestCase {
  id: string;
  input: string | string[];
  expected_output?: string;
  context?: string[];
  retrieval_context?: string[];
  tools_called?: string[];
  expected_tools?: string[];
  graders: (string | Grader)[];
  metadata?: Record<string, unknown>;
  tags?: string[];
  provider?: ProviderConfig;
  params?: Record<string, unknown>;
  timeout_ms?: number;
  weight?: number;
}

export interface EvalSuite {
  $schema?: string;
  version: string;
  id: string;
  name?: string;
  description?: string;
  graders?: Grader[];
  test_cases?: TestCase[];
  test_cases_file?: string;
  config?: SuiteConfig;
  metadata?: Record<string, unknown>;
  tags?: string[];
}

export interface GraderResult {
  grader_id: string;
  type: string;
  score: number | null;
  passed: boolean;
  reason?: string;
  metadata?: Record<string, unknown>;
}

export interface Result {
  test_case_id: string;
  actual_output?: string;
  grader_results: GraderResult[];
  passed: boolean;
  duration_ms?: number;
  completed_at?: string;
  error?: {
    type: "timeout" | "provider_error" | "runner_error";
    message?: string;
    code?: string | number;
    retryable?: boolean;
  };
  metadata?: Record<string, unknown>;
}

export interface SummaryByGrader {
  passed: number;
  failed: number;
  avg_score: number;
}

export interface Summary {
  total?: number;
  passed?: number;
  failed?: number;
  skipped?: number;
  pass_rate?: number;
  avg_score?: number;
  duration_ms?: number;
  by_grader?: Record<string, SummaryByGrader>;
}

export interface ResultSet {
  $schema?: string;
  version: string;
  suite_id: string;
  suite_version?: string;
  run_id: string;
  started_at: string;
  completed_at?: string;
  provider?: ProviderConfig;
  runner?: { name: string; version: string };
  results: Result[];
  summary?: Summary;
  metadata?: Record<string, unknown>;
}

export type DocumentType = "testcase" | "grader" | "suite" | "resultset";

export interface ValidationError {
  path: string;
  message: string;
  code: string;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
}