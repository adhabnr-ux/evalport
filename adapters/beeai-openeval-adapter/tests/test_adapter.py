import json

import pytest
from openeval.validate import validate_result_set, validate_suite

from beeai_openeval_adapter import (
    from_openeval,
    result_to_openeval,
    results_to_openeval,
    to_openeval,
)

try:  # real-object tests below skip cleanly when the framework isn't installed
    import beeai_framework  # noqa: F401

    HAS_BEEAI = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    HAS_BEEAI = False

requires_beeai = pytest.mark.skipif(not HAS_BEEAI, reason="beeai-framework not installed")


# The exact shape of python/examples/evaluation/dataset.json in
# i-am-bee/beeai-framework (loaded by examples/evaluation/dataset.py).
DATASET_ITEMS = [
    {
        "question": "Which magazine was started first Arthur's Magazine or First for Women?",
        "expected_answer": "Arthur's Magazine",
        "supporting_sentences": [
            "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.",
            "First for Women is a woman's magazine published by Bauer Media Group in the USA.",
        ],
        "expected_tool_calls": 2,
        "supporting_titles": ["Arthur's Magazine", "First for Women"],
    },
    {
        "question": "The Oberoi family is part of a hotel company that has a head office in what city?",
        "expected_answer": "Delhi",
        "supporting_sentences": [
            "The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.",
            "The Oberoi Group is a hotel company with its head office in Delhi.",
        ],
        "expected_tool_calls": 2,
        "supporting_titles": ["Oberoi family", "The Oberoi Group"],
    },
]


def _plain_message(text, role="assistant"):
    """A Message.to_plain()-shaped dict, for the duck-typed path."""
    return {"role": role, "content": [{"type": "text", "text": text}]}


def _plain_tool_call(tool_name, args, call_id="c1"):
    return {"role": "assistant", "content": [{"type": "tool-call", "id": call_id, "tool_name": tool_name, "args": json.dumps(args)}]}


def _plain_agent_output(text, structured=None, context=None):
    """An AgentOutput-shaped dict: output (messages), context, output_structured."""
    return {"output": [_plain_message(text)], "context": context or {}, "output_structured": structured}


# ---------------------------------------------------------------------------
# dataset -> Suite
# ---------------------------------------------------------------------------


def test_to_openeval_maps_dataset_json_shape():
    suite = to_openeval(DATASET_ITEMS, suite_id="beeai_rag_multi_hop")

    assert suite["id"] == "beeai_rag_multi_hop"
    assert suite["metadata"]["openeval"]["source"] == "beeai-framework"
    assert len(suite["test_cases"]) == 2

    tc = suite["test_cases"][1]
    assert tc["id"] == "tc_1"
    assert tc["input"] == DATASET_ITEMS[1]["question"]
    assert tc["expected_output"] == "Delhi"
    assert tc["context"] == DATASET_ITEMS[1]["supporting_sentences"]
    assert tc["graders"] == ["gr_beeai_exact_match"]
    assert tc["metadata"]["beeai"]["expected_tool_calls"] == 2
    assert tc["metadata"]["beeai"]["supporting_titles"] == ["Oberoi family", "The Oberoi Group"]
    # The dataset never names the tool, so expected_tools is not invented.
    assert "expected_tools" not in tc

    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"gr_beeai_exact_match"}
    assert suite["graders"][0]["type"] == "exact_match"


def test_to_openeval_expected_tool_expands_count_into_expected_tools():
    suite = to_openeval(DATASET_ITEMS, suite_id="s", expected_tool="Wikipedia")
    assert suite["test_cases"][0]["expected_tools"] == ["Wikipedia", "Wikipedia"]
    assert validate_suite(suite).valid, validate_suite(suite).errors


def test_to_openeval_preserves_unknown_keys_and_ids_losslessly():
    item = dict(DATASET_ITEMS[0], id="hotpot_0001", difficulty="hard", split="validation")
    suite = to_openeval([item], suite_id="s")
    tc = suite["test_cases"][0]
    assert tc["id"] == "hotpot_0001"
    assert tc["metadata"]["beeai"]["extra"] == {"difficulty": "hard", "split": "validation"}


def test_to_openeval_list_of_acceptable_answers():
    item = dict(DATASET_ITEMS[0], expected_answer=["Arthur's Magazine", "Arthur's"])
    suite = to_openeval([item], suite_id="s")
    tc = suite["test_cases"][0]
    assert tc["expected_output"] == "Arthur's Magazine"
    assert tc["metadata"]["beeai"]["acceptable_answers"] == ["Arthur's Magazine", "Arthur's"]
    assert validate_suite(suite).valid, validate_suite(suite).errors


def test_to_openeval_accepts_golden_style_aliases():
    # The DeepEval Golden shape rag_goldens() maps dataset items onto.
    goldens = [{"input": "Q?", "expected_output": "A", "context": ["ctx"], "expected_tools": [{"name": "Wikipedia"}]}]
    suite = to_openeval(goldens, suite_id="s")
    tc = suite["test_cases"][0]
    assert tc["input"] == "Q?"
    assert tc["expected_output"] == "A"
    assert tc["context"] == ["ctx"]
    assert tc["expected_tools"] == ["Wikipedia"]
    assert validate_suite(suite).valid, validate_suite(suite).errors


def test_to_openeval_validates_against_evalport_spec():
    suite = to_openeval(DATASET_ITEMS, suite_id="beeai_rag_multi_hop", name="RAG multi-hop")
    validation = validate_suite(suite)
    assert validation.valid, validation.errors
    json.dumps(suite)  # fully serializable


def test_to_openeval_rejects_empty_and_questionless_items():
    with pytest.raises(ValueError):
        to_openeval([])
    with pytest.raises(ValueError):
        to_openeval([{"expected_answer": "x"}])


def test_from_openeval_round_trip_matches_dataset_json():
    suite = to_openeval(DATASET_ITEMS, suite_id="s")
    items = from_openeval(suite)
    assert len(items) == 2
    for original, recovered in zip(DATASET_ITEMS, items):
        for key in ("question", "expected_answer", "supporting_sentences", "expected_tool_calls", "supporting_titles"):
            assert recovered[key] == original[key], key
    # An EvalPort suite re-imported through to_openeval() is stable.
    again = to_openeval(items, suite_id="s")
    assert [tc["input"] for tc in again["test_cases"]] == [tc["input"] for tc in suite["test_cases"]]
    assert [tc["id"] for tc in again["test_cases"]] == [tc["id"] for tc in suite["test_cases"]]


def test_from_openeval_converts_foreign_suites():
    """Importing any EvalPort suite into BeeAI's dataset shape is the point of
    this direction, so foreign test cases are converted, not skipped."""
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {"id": "tc1", "input": "hi", "expected_output": "hello", "graders": ["g1"], "expected_tools": ["Wikipedia", "Weather"]}
        ],
    }
    items = from_openeval(suite)
    assert items == [
        {
            "id": "tc1",
            "question": "hi",
            "expected_answer": "hello",
            "supporting_sentences": [],
            "expected_tool_calls": 2,
            "supporting_titles": [],
            "expected_tools": ["Wikipedia", "Weather"],
        }
    ]


# ---------------------------------------------------------------------------
# AgentOutput -> Result / ResultSet (duck-typed)
# ---------------------------------------------------------------------------


def test_result_to_openeval_exact_match_from_plain_output():
    output = _plain_agent_output("Delhi", structured={"answer": "Delhi"}, context={"trace_id": "abc"})
    result = result_to_openeval("tc_1", output, expected_output="Delhi", duration_ms=1234.6)

    assert result["test_case_id"] == "tc_1"
    assert result["actual_output"] == "Delhi"
    assert result["passed"] is True
    assert result["duration_ms"] == 1235
    gr = result["grader_results"][0]
    assert gr == {"grader_id": "gr_beeai_exact_match", "type": "exact_match", "score": 1.0, "passed": True}
    beeai = result["metadata"]["beeai"]
    assert beeai["output"] == [_plain_message("Delhi")]
    assert beeai["context"] == {"trace_id": "abc"}
    assert beeai["output_structured"] == {"answer": "Delhi"}


def test_result_to_openeval_wrong_answer_fails():
    result = result_to_openeval("tc_1", _plain_agent_output("Mumbai"), expected_output="Delhi")
    assert result["passed"] is False
    assert result["grader_results"][0]["score"] == 0.0


def test_result_to_openeval_empty_output_list_falls_back_to_structured_then_empty():
    # Empty message list but a structured answer: actual_output is its JSON.
    result = result_to_openeval("tc_1", {"output": [], "output_structured": {"answer": "Delhi"}}, expected_output="Delhi")
    assert result["actual_output"] == '{"answer": "Delhi"}'
    assert result["metadata"]["beeai"]["output"] == []
    assert result["passed"] is False  # JSON blob != "Delhi", and that's honest

    # Nothing at all: RunnableOutput.last_message falls back to AssistantMessage("").
    result = result_to_openeval("tc_1", {"output": []}, expected_output="Delhi")
    assert result["actual_output"] == ""
    assert result["passed"] is False


def test_result_to_openeval_unscored_when_nothing_to_grade_with():
    result = result_to_openeval("tc_1", _plain_agent_output("Delhi"))
    assert result["passed"] is False
    gr = result["grader_results"][0]
    assert gr["grader_id"] == "gr_beeai_unscored"
    assert gr["score"] is None
    assert gr["passed"] is False
    assert "error" not in result  # unscored is not an error


def test_result_to_openeval_passes_external_metric_results_through():
    # DeepEval MetricData-like objects and ragas MetricResult-like dicts.
    class MetricData:
        def __init__(self, name, score, success, threshold, reason=None):
            self.name, self.score, self.success, self.threshold, self.reason = name, score, success, threshold, reason

    graders = [
        MetricData("Exact Match", 1.0, True, 1.0),
        MetricData("Answer Relevancy", 0.62, False, 0.7, reason="partially relevant"),
        {"name": "ToolCallAccuracy", "value": 0.5},
        {"grader_id": "gr_custom", "type": "llm_judge", "score": 0.9, "passed": True},
    ]
    result = result_to_openeval("tc_1", _plain_agent_output("Delhi"), grader_results=graders)
    ids = [g["grader_id"] for g in result["grader_results"]]
    assert ids == ["gr_beeai_exact_match", "gr_beeai_answer_relevancy", "gr_beeai_toolcallaccuracy", "gr_custom"]
    assert result["grader_results"][1]["reason"] == "partially relevant"
    assert result["grader_results"][1]["metadata"] == {"name": "Answer Relevancy", "threshold": 0.7}
    # value-only metric with no threshold: passed only on a perfect score
    assert result["grader_results"][2]["score"] == 0.5
    assert result["grader_results"][2]["passed"] is False
    assert result["grader_results"][3]["type"] == "llm_judge"
    assert result["passed"] is False  # one metric failed


def test_result_to_openeval_clamps_out_of_range_scores_and_keeps_raw():
    result = result_to_openeval("tc_1", _plain_agent_output("x"), grader_results=[{"name": "latency", "score": 7.5, "passed": True}])
    gr = result["grader_results"][0]
    assert gr["score"] == 1.0
    assert gr["metadata"]["raw_score"] == 7.5


def test_result_to_openeval_error_path():
    class ChatModelError(Exception):
        pass

    try:
        raise ChatModelError("provider returned 500")
    except ChatModelError as exc:
        result = result_to_openeval("tc_1", None, expected_output="Delhi", error=exc, duration_ms=50)

    assert result["passed"] is False
    assert "actual_output" not in result
    assert result["error"] == {"type": "provider_error", "message": "provider returned 500"}
    assert result["metadata"]["beeai"]["error"]["error_class"] == "ChatModelError"
    gr = result["grader_results"][0]
    assert gr["grader_id"] == "gr_beeai_exact_match"
    assert gr["score"] is None and gr["passed"] is False

    timeout = result_to_openeval("tc_2", None, error=TimeoutError("agent exceeded 60s"))
    assert timeout["error"]["type"] == "timeout"
    assert timeout["grader_results"][0]["grader_id"] == "gr_beeai_unscored"

    generic = result_to_openeval("tc_3", None, error=RuntimeError("boom"))
    assert generic["error"]["type"] == "runner_error"

    as_dict = result_to_openeval("tc_4", None, error={"type": "provider_error", "message": "429", "code": 429, "retryable": True})
    assert as_dict["error"] == {"type": "provider_error", "message": "429", "code": 429, "retryable": True}


def test_result_to_openeval_extracts_tool_calls_from_plain_trajectory():
    output = {
        "output": [_plain_message("Delhi")],
        "state": {
            "iteration": 2,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "memory": {
                "messages": [
                    _plain_message("Which city?", role="user"),
                    _plain_tool_call("Wikipedia", {"query": "Oberoi family"}),
                    {"role": "tool", "content": [{"type": "tool-result", "result": "...", "tool_name": "Wikipedia", "tool_call_id": "c1"}]},
                    _plain_message("Delhi"),
                ]
            },
        },
    }
    result = result_to_openeval("tc_1", output, expected_output="Delhi")
    beeai = result["metadata"]["beeai"]
    assert beeai["tools_called"] == ["Wikipedia"]
    assert beeai["tool_calls"] == [{"id": "c1", "tool_name": "Wikipedia", "args": {"query": "Oberoi family"}}]
    assert beeai["state"]["iteration"] == 2
    assert beeai["state"]["usage"]["total_tokens"] == 15
    assert len(beeai["state"]["trajectory"]) == 4


def test_results_to_openeval_builds_summary_and_validates():
    runs = [
        ("tc_0", _plain_agent_output("Arthur's Magazine")),  # unscored tuple form
        {"test_case_id": "tc_1", "output": _plain_agent_output("Delhi"), "expected_output": "Delhi", "duration_ms": 10},
        result_to_openeval("tc_2", _plain_agent_output("Mumbai"), expected_output="Delhi", duration_ms=20),
    ]
    result_set = results_to_openeval(
        runs,
        suite_id="beeai_rag_multi_hop",
        run_id="run_2026_09_27",
        started_at="2026-09-27T00:00:00Z",
        completed_at="2026-09-27T00:01:00Z",
        model="ollama:llama3.1:8b",
        isolation="fresh",
    )
    assert result_set["suite_id"] == "beeai_rag_multi_hop"
    assert result_set["provider"] == {"model": "ollama:llama3.1:8b"}
    assert result_set["runner"]["name"] == "beeai-framework"
    assert result_set["isolation"] == "fresh"
    assert result_set["metadata"]["openeval"]["source"] == "beeai-framework"
    summary = result_set["summary"]
    assert summary["total"] == 3
    assert summary["passed"] == 1
    assert summary["failed"] == 2
    assert summary["pass_rate"] == pytest.approx(1 / 3)
    assert summary["avg_score"] == pytest.approx(0.5)  # null score excluded
    assert "duration_ms" not in summary  # not every result had one
    assert summary["by_grader"]["gr_beeai_exact_match"] == {"passed": 1, "failed": 1, "avg_score": 0.5}
    assert summary["by_grader"]["gr_beeai_unscored"] == {"passed": 0, "failed": 1}

    validation = validate_result_set(result_set)
    assert validation.valid, validation.errors
    json.dumps(result_set)


def test_results_to_openeval_rejects_empty_and_malformed():
    with pytest.raises(ValueError):
        results_to_openeval([], suite_id="s", run_id="r", started_at="2026-09-27T00:00:00Z")
    with pytest.raises(ValueError):
        results_to_openeval([{"output": {}}], suite_id="s", run_id="r", started_at="2026-09-27T00:00:00Z")
    with pytest.raises(ValueError):
        results_to_openeval(["tc_1"], suite_id="s", run_id="r", started_at="2026-09-27T00:00:00Z")


def test_full_round_trip_suite_and_results_share_ids():
    """The suite built by to_openeval() and the ResultSet built by
    results_to_openeval() must reference the same TestCase ids, since a real
    consumer joins them to compute pass rates per test case."""
    suite = to_openeval(DATASET_ITEMS, suite_id="beeai_rag_multi_hop")
    suite_ids = [tc["id"] for tc in suite["test_cases"]]

    # What a BeeAI experiment loop does: for each item, run the agent, keep the id.
    runs = []
    for tc, answer in zip(suite["test_cases"], ["Arthur's Magazine", "Delhi"]):
        runs.append({"test_case_id": tc["id"], "output": _plain_agent_output(answer), "expected_output": tc["expected_output"]})
    result_set = results_to_openeval(runs, suite_id=suite["id"], run_id="run1", started_at="2026-09-27T00:00:00Z")

    assert [r["test_case_id"] for r in result_set["results"]] == suite_ids
    assert result_set["suite_id"] == suite["id"]
    assert result_set["summary"]["pass_rate"] == 1.0
    assert validate_suite(suite).valid, validate_suite(suite).errors
    assert validate_result_set(result_set).valid, validate_result_set(result_set).errors


# ---------------------------------------------------------------------------
# real beeai-framework objects, constructed offline (no LLM, no network)
# ---------------------------------------------------------------------------


@requires_beeai
def test_real_agent_output_with_structured_output():
    from beeai_framework.agents import AgentOutput
    from beeai_framework.backend import AssistantMessage
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str
        supporting_titles: list

    output = AgentOutput(
        output=[AssistantMessage('{"answer": "Delhi"}')],
        output_structured=Answer(answer="Delhi", supporting_titles=["The Oberoi Group"]),
        context={"run": "x"},
    )
    result = result_to_openeval("tc_1", output, expected_output='{"answer": "Delhi"}')
    assert result["actual_output"] == '{"answer": "Delhi"}'
    assert result["passed"] is True
    beeai = result["metadata"]["beeai"]
    assert beeai["output"] == [{"role": "assistant", "content": [{"type": "text", "text": '{"answer": "Delhi"}'}]}]
    assert beeai["output_structured"] == {"answer": "Delhi", "supporting_titles": ["The Oberoi Group"]}
    assert beeai["context"] == {"run": "x"}
    json.dumps(result)


@requires_beeai
def test_real_agent_output_empty_message_list():
    from beeai_framework.agents import AgentOutput

    # RunnableOutput.last_message falls back to AssistantMessage("") here.
    result = result_to_openeval("tc_1", AgentOutput(output=[]), expected_output="Delhi")
    assert result["actual_output"] == ""
    assert result["passed"] is False
    assert result["metadata"]["beeai"]["output"] == []


@requires_beeai
def test_real_requirement_agent_output_shape():
    """Builds the exact object RequirementAgent.run() returns
    (RequirementAgentOutput(output=[state.answer], output_structured=state.result,
    state=final_state), see beeai_framework/agents/requirement/agent.py)
    without running an agent: the state's memory holds the trajectory a
    multi-hop run through WikipediaTool produces."""
    import asyncio

    from beeai_framework.agents.requirement.types import RequirementAgentOutput, RequirementAgentRunState
    from beeai_framework.backend import AssistantMessage, ToolMessage, UserMessage
    from beeai_framework.backend.message import MessageToolCallContent, MessageToolResultContent
    from beeai_framework.backend.types import ChatModelCost, ChatModelUsage
    from beeai_framework.memory import UnconstrainedMemory

    memory = UnconstrainedMemory()

    async def fill() -> None:
        await memory.add_many(
            [
                UserMessage("The Oberoi family is part of a hotel company that has a head office in what city?"),
                AssistantMessage(MessageToolCallContent(id="call_1", tool_name="Wikipedia", args=json.dumps({"query": "Oberoi family"}))),
                ToolMessage(MessageToolResultContent(result="The Oberoi family ... The Oberoi Group.", tool_name="Wikipedia", tool_call_id="call_1")),
                AssistantMessage(MessageToolCallContent(id="call_2", tool_name="Wikipedia", args=json.dumps({"query": "The Oberoi Group"}))),
                ToolMessage(MessageToolResultContent(result="... head office in Delhi.", tool_name="Wikipedia", tool_call_id="call_2")),
                AssistantMessage("Delhi"),
            ]
        )

    asyncio.run(fill())
    answer = AssistantMessage("Delhi")
    state = RequirementAgentRunState(
        answer=answer,
        result={"answer": "Delhi"},
        memory=memory,
        iteration=3,
        steps=[],
        usage=ChatModelUsage(prompt_tokens=120, completion_tokens=30, total_tokens=150),
        cost=ChatModelCost(total_cost_usd=0.001),
    )
    output = RequirementAgentOutput(output=[state.answer], output_structured=state.result, state=state)

    suite = to_openeval(DATASET_ITEMS, suite_id="beeai_rag_multi_hop", expected_tool="Wikipedia")
    tc = suite["test_cases"][1]
    result = result_to_openeval(tc["id"], output, expected_output=tc["expected_output"], duration_ms=812)

    assert result["actual_output"] == "Delhi"
    assert result["passed"] is True
    beeai = result["metadata"]["beeai"]
    assert beeai["tools_called"] == ["Wikipedia", "Wikipedia"]
    assert beeai["tools_called"] == tc["expected_tools"]
    assert beeai["tool_calls"][0] == {"id": "call_1", "tool_name": "Wikipedia", "args": {"query": "Oberoi family"}}
    assert beeai["state"]["iteration"] == 3
    assert beeai["state"]["usage"]["total_tokens"] == 150
    assert beeai["state"]["cost"]["total_cost_usd"] == 0.001
    assert len(beeai["state"]["trajectory"]) == 6
    assert beeai["state"]["trajectory"][0]["role"] == "user"

    result_set = results_to_openeval([result], suite_id=suite["id"], run_id="run1", started_at="2026-09-27T00:00:00Z", model="ollama:llama3.1:8b")
    assert result_set["runner"]["version"]  # installed beeai-framework version via importlib.metadata
    assert result_set["summary"]["duration_ms"] == 812
    assert validate_suite(suite).valid, validate_suite(suite).errors
    assert validate_result_set(result_set).valid, validate_result_set(result_set).errors
    json.dumps(result_set)


@requires_beeai
def test_real_framework_error_maps_to_provider_error():
    from beeai_framework.backend.errors import ChatModelError

    result = result_to_openeval("tc_1", None, expected_output="Delhi", error=ChatModelError("upstream 503"))
    assert result["error"]["type"] == "provider_error"
    assert result["metadata"]["beeai"]["error"]["error_class"] == "ChatModelError"
    assert "explain" in result["metadata"]["beeai"]["error"]
    assert validate_result_set(
        results_to_openeval([result], suite_id="s", run_id="r", started_at="2026-09-27T00:00:00Z")
    ).valid


@requires_beeai
def test_real_requirement_agent_run_offline():
    """Actually runs a RequirementAgent (the agent i-am-bee/beeai-framework#1677
    asked this adapter to be tested against) against a scripted ChatModel and a
    local @tool -- no LLM, no network (socket.connect is patched to fail) --
    then converts the RequirementAgentOutput it returns. The scripted-model
    pattern mirrors beeai's own python/tests/examples/test_agent_guild_preflight.py."""
    import asyncio
    import socket
    from collections.abc import AsyncGenerator

    from beeai_framework.agents.requirement import RequirementAgent
    from beeai_framework.agents.requirement.types import RequirementAgentOutput
    from beeai_framework.backend import AssistantMessage
    from beeai_framework.backend.chat import ChatModel
    from beeai_framework.backend.message import MessageToolCallContent
    from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
    from beeai_framework.context import RunContext
    from beeai_framework.memory import UnconstrainedMemory
    from beeai_framework.tools import StringToolOutput, tool

    @tool
    def wikipedia(query: str) -> StringToolOutput:
        """Look up a Wikipedia article."""
        return StringToolOutput("The Oberoi Group is a hotel company with its head office in Delhi.")

    class ScriptedModel(ChatModel):
        def __init__(self) -> None:
            super().__init__(
                allow_parallel_tool_calls=True,
                retry_on_empty_response=False,
                fix_invalid_tool_calls=False,
                tool_call_fallback_via_response_format=False,
            )
            self.calls = 0

        @property
        def model_id(self) -> str:
            return "synthetic-local-script"

        @property
        def provider_id(self):
            return "openai"  # a namespace only; no provider adapter is constructed

        async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
            self.calls += 1
            name, args = ("wikipedia", {"query": "Oberoi"}) if self.calls == 1 else ("final_answer", {"response": "Delhi"})
            return ChatModelOutput(
                output=[AssistantMessage(MessageToolCallContent(id=str(self.calls), tool_name=name, args=json.dumps(args)))]
            )

        async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
            yield await self._create(input, run)

    def refuse(*args, **kwargs):
        raise AssertionError("Tests must not open a network connection")

    original_connect, original_connect_ex = socket.socket.connect, socket.socket.connect_ex
    socket.socket.connect, socket.socket.connect_ex = refuse, refuse
    try:
        suite = to_openeval(DATASET_ITEMS, suite_id="beeai_rag_multi_hop", expected_tool="wikipedia")
        tc = suite["test_cases"][1]
        agent = RequirementAgent(llm=ScriptedModel(), tools=[wikipedia], memory=UnconstrainedMemory())

        async def run_agent() -> RequirementAgentOutput:
            return await agent.run(tc["input"])  # agent.run() returns an awaitable Run, not a coroutine

        output = asyncio.run(run_agent())
    finally:
        socket.socket.connect, socket.socket.connect_ex = original_connect, original_connect_ex

    assert isinstance(output, RequirementAgentOutput)
    result = result_to_openeval(tc["id"], output, expected_output=tc["expected_output"], duration_ms=5)
    assert result["actual_output"] == "Delhi"
    assert result["passed"] is True
    beeai = result["metadata"]["beeai"]
    # final_answer is RequirementAgent's internal tool: excluded from tools_called
    # like BeeAI's own count_tool_usage(), but kept in the raw tool_calls list.
    assert beeai["tools_called"] == ["wikipedia"]
    assert [c["tool_name"] for c in beeai["tool_calls"]] == ["wikipedia", "final_answer"]
    assert beeai["tool_calls"][0]["args"] == {"query": "Oberoi"}
    assert [s["tool"] for s in beeai["state"]["steps"]] == ["wikipedia", "final_answer"]
    assert beeai["state"]["iteration"] == 2
    assert beeai["state"]["trajectory"][0]["role"] == "user"
    assert beeai["output_structured"] is not None

    result_set = results_to_openeval(
        [result], suite_id=suite["id"], run_id="run1", started_at="2026-09-27T00:00:00Z", model=agent._llm.model_id
    )
    assert result_set["provider"] == {"model": "synthetic-local-script"}
    assert validate_result_set(result_set).valid, validate_result_set(result_set).errors
    json.dumps(result_set)
