from openeval.validate import validate_suite

from crewai_openeval_adapter import to_openeval, from_openeval


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeAgent:
    def __init__(self, role, tools=None):
        self.role = role
        self.tools = tools or []


class FakeTaskOutput:
    """Stand-in for CrewAI's TaskOutput class (attribute-based)."""

    def __init__(self, id, description, expected_output=None, tools=None, agent=None):
        self.id = id
        self.description = description
        self.expected_output = expected_output
        self.tools = tools
        self.agent = agent


class FakeCrewOutput:
    def __init__(self, id, tasks_output):
        self.id = id
        self.tasks_output = tasks_output


def test_to_openeval_from_objects():
    result = FakeCrewOutput(
        id="run1",
        tasks_output=[
            FakeTaskOutput(
                "t1",
                "Find the top 3 competitors",
                expected_output="A list of 3 competitor names",
                tools=[FakeTool("search_tool")],
                agent=FakeAgent("researcher"),
            ),
            FakeTaskOutput("t2", "Summarize into a memo", expected_output="A 3-paragraph memo"),
        ],
    )
    suite = to_openeval(result)

    assert suite["id"] == "crewai_eval_run1"
    assert len(suite["test_cases"]) == 2

    tc1 = suite["test_cases"][0]
    assert tc1["expected_tools"] == ["search_tool"]
    assert tc1["expected_output"] == "A list of 3 competitor names"
    assert tc1["metadata"]["crewai_agent"] == "researcher"
    assert set(tc1["graders"]) == {"gr_output_match", "gr_tool_selection"}

    tc2 = suite["test_cases"][1]
    assert "expected_tools" not in tc2
    assert tc2["graders"] == ["gr_output_match"]

    grader_ids = {g["id"] for g in suite["graders"]}
    assert grader_ids == {"gr_output_match", "gr_tool_selection"}


def test_to_openeval_from_dicts():
    result = {
        "run_id": "run2",
        "tasks": [{"id": "t1", "description": "hi", "expected_output": "hello"}],
    }
    suite = to_openeval(result)
    assert suite["test_cases"][0]["expected_output"] == "hello"
    assert suite["test_cases"][0]["input"] == "hi"


def test_to_openeval_validates_against_evalport_spec():
    result = FakeCrewOutput(
        id="run3",
        tasks_output=[FakeTaskOutput("t1", "task", expected_output="out", tools=[FakeTool("x")])],
    )
    suite = to_openeval(result)
    validation = validate_suite(suite)
    assert validation.valid, validation.errors


def test_default_grader_is_llm_judge():
    result = FakeCrewOutput(id="run4", tasks_output=[FakeTaskOutput("t1", "task", expected_output="out")])
    suite = to_openeval(result)
    assert suite["graders"][0]["type"] == "llm_judge"
    assert "{output}" in suite["graders"][0]["params"]["prompt"]


def test_exact_match_grader_option():
    result = FakeCrewOutput(id="run5", tasks_output=[FakeTaskOutput("t1", "task", expected_output="out")])
    suite = to_openeval(result, grader_type="exact_match")
    assert suite["graders"][0]["type"] == "exact_match"


def test_from_openeval_round_trip():
    suite = {
        "version": "1.0.0",
        "id": "s1",
        "graders": [{"id": "g1", "type": "exact_match"}],
        "test_cases": [
            {"id": "tc1", "input": "hi", "expected_output": "hello", "expected_tools": ["t"], "graders": ["g1"]},
        ],
    }
    tasks = from_openeval(suite)
    assert tasks == [
        {"id": "tc1", "description": "hi", "expected_output": "hello", "tools": ["t"]},
    ]


def test_empty_results_still_valid_shape():
    result = FakeCrewOutput(id="run6", tasks_output=[])
    suite = to_openeval(result)
    assert suite["test_cases"] == []
    assert suite["version"] == "1.0.0"
