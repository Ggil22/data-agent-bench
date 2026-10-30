import shutil

import pytest
from conftest import ScriptedLLM, call

from data_agent_bench.agent import run_agent
from data_agent_bench.bench.failures import classify
from data_agent_bench.bench.report import render_markdown, wilson_interval
from data_agent_bench.bench.runner import run_benchmark
from data_agent_bench.bench.scoring import score
from data_agent_bench.bench.tasks import TASKS_BY_ID
from data_agent_bench.llm import LLMResponse
from data_agent_bench.tools import ToolExecutor


@pytest.fixture
def executor(data_dir, tmp_path):
    for f in data_dir.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    return ToolExecutor(tmp_path)


def evaluate(task_id, llm, executor, data, max_steps=12):
    task = TASKS_BY_ID[task_id]
    traj = run_agent(llm, task.question, executor, max_steps=max_steps)
    sc = score(task, task.compute(data), traj.final_answer)
    return traj, classify(traj, sc, task.answer_type == "unanswerable")


def test_successful_episode(executor, data):
    llm = ScriptedLLM(
        [
            call("run_python", code="import pandas as pd\nlen(pd.read_csv('sales.csv'))"),
            call("final_answer", answer=str(len(data["sales"]))),
        ]
    )
    traj, (outcome, flags) = evaluate("agg_001", llm, executor, data)
    assert traj.stop_reason == "final_answer"
    assert outcome == "correct"
    assert flags == []
    # the tool output was fed back to the model
    tool_msg = llm.seen_messages[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"] == str(len(data["sales"]))


def test_hallucinated_tool_is_flagged(executor, data):
    llm = ScriptedLLM([call("sql_query", query="SELECT 1"), call("final_answer", answer="1")])
    _, (outcome, flags) = evaluate("agg_001", llm, executor, data)
    assert "tool_hallucination" in flags
    assert outcome == "answered_without_evidence"


def test_loop_is_detected(executor, data):
    llm = ScriptedLLM([call("list_files")] * 5)
    traj, (outcome, flags) = evaluate("agg_001", llm, executor, data, max_steps=5)
    assert traj.stop_reason == "max_steps"
    assert outcome == "loop"
    assert "repeated_call" in flags


def test_text_only_turn_gets_nudged(executor, data):
    llm = ScriptedLLM([LLMResponse(content="Let me think."), call("final_answer", answer="x")])
    traj, (_, flags) = evaluate("agg_001", llm, executor, data)
    assert "text_only_turn" in flags
    assert llm.seen_messages[1][-1]["role"] == "user"


def test_hallucinated_answer_on_unanswerable_task(executor, data):
    llm = ScriptedLLM(
        [call("inspect_csv", filename="customers.csv"), call("final_answer", answer="41")]
    )
    _, (outcome, _) = evaluate("un_001", llm, executor, data)
    assert outcome == "hallucinated_answer"


def test_api_error_ends_episode(executor, data):
    class FailingLLM:
        def chat(self, messages, tools):
            raise ConnectionError("rate limited")

    traj, (outcome, _) = evaluate("agg_001", FailingLLM(), executor, data)
    assert traj.stop_reason == "api_error" and outcome == "api_error"


def test_runner_end_to_end_and_resume(data_dir, data, tmp_path):
    tasks = [TASKS_BY_ID["agg_001"], TASKS_BY_ID["un_001"]]
    answers = {"agg_001": str(len(data["sales"])), "un_001": "UNANSWERABLE"}

    def make_client(model, repeat):
        class Oracle:
            def chat(self, messages, tools):
                question = messages[1]["content"]
                task = next(t for t in tasks if t.question == question)
                return call("final_answer", answer=answers[task.id])

        return Oracle()

    out = tmp_path / "run"
    results = run_benchmark(["oracle"], tasks, make_client, data_dir, out, log=lambda _: None)
    lines = results.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    run_benchmark(["oracle"], tasks, make_client, data_dir, out, log=lambda _: None)
    assert len(results.read_text(encoding="utf-8").splitlines()) == 2  # resumed, nothing re-run

    import json

    report = render_markdown([json.loads(line) for line in lines], "test")
    assert "| oracle | 2 | 100.0%" in report


def test_wilson_interval():
    lo, hi = wilson_interval(8, 10)
    assert 0.49 < lo < 0.5 and 0.94 < hi < 0.95
    assert wilson_interval(0, 0) == (0.0, 0.0)
