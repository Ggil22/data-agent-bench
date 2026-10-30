"""The agent loop: an LLM calling tools until it submits a final answer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from data_agent_bench.llm import LLMClient, ToolCall
from data_agent_bench.tools import TOOL_SPECS, ToolExecutor, ToolResult

SYSTEM_PROMPT = """\
You are a data analyst agent. You answer questions about the data files in your working directory.

Tools:
- list_files: see which files are available.
- inspect_csv: see the schema and first rows of a CSV file.
- run_python: run Python code (pandas and numpy are available). Every call runs in a fresh \
process, so re-load any data you need. Use print() to show results; the value of the last \
expression is printed too.
- final_answer: submit your answer. This ends the task.

Rules:
- Base every answer on code you actually ran on the data. Never guess numbers.
- If a column or value is ambiguous, look for documentation files in the working directory.
- Call final_answer with ONLY the answer value: no units, no explanation, no thousand separators.
  - Numbers: plain decimal number, rounded as the question asks.
  - Lists: comma-separated values, in the order the question asks for.
  - Yes/no questions: yes or no.
- If the question cannot be answered with the available data, call final_answer with exactly: \
UNANSWERABLE
"""

FINAL_ANSWER_SPEC = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit the final answer to the question. This ends the task.",
        "parameters": {
            "type": "object",
            "properties": {"answer": {"type": "string", "description": "Only the answer value."}},
            "required": ["answer"],
        },
    },
}

AGENT_TOOL_SPECS = [*TOOL_SPECS, FINAL_ANSWER_SPEC]

NUDGE = "Use the tools to work on the task. When you are done, call final_answer."


@dataclass
class ToolExchange:
    name: str
    arguments: str
    output: str
    is_error: bool = False
    error_type: str | None = None


@dataclass
class Step:
    content: str | None
    tool_calls: list[ToolExchange]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


@dataclass
class Trajectory:
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    # final_answer | max_steps | api_error
    stop_reason: str = "max_steps"
    error: str | None = None

    @property
    def prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    @property
    def exchanges(self) -> list[ToolExchange]:
        return [ex for s in self.steps for ex in s.tool_calls]


def run_agent(
    llm: LLMClient, question: str, executor: ToolExecutor, max_steps: int = 12
) -> Trajectory:
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    traj = Trajectory()

    for _ in range(max_steps):
        try:
            response = llm.chat(messages, AGENT_TOOL_SPECS)
        except Exception as exc:  # any provider failure ends the episode
            traj.stop_reason = "api_error"
            traj.error = f"{type(exc).__name__}: {exc}"
            return traj

        step = Step(
            content=response.content,
            tool_calls=[],
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=response.latency_s,
            cached=response.cached,
        )
        traj.steps.append(step)

        if not response.tool_calls:
            messages.append({"role": "assistant", "content": response.content or ""})
            messages.append({"role": "user", "content": NUDGE})
            continue

        messages.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [_tool_call_message(tc) for tc in response.tool_calls],
            }
        )
        for tc in response.tool_calls:
            if tc.name == "final_answer":
                answer = _parse_final_answer(tc.arguments)
                if answer is not None:
                    step.tool_calls.append(ToolExchange(tc.name, tc.arguments, ""))
                    traj.final_answer = answer
                    traj.stop_reason = "final_answer"
                    return traj
                result = ToolResult(
                    "Error: final_answer needs a JSON object with an 'answer' field.",
                    True,
                    "invalid_arguments",
                )
            else:
                result = executor.execute(tc.name, tc.arguments)
            step.tool_calls.append(
                ToolExchange(
                    tc.name, tc.arguments, result.output, result.is_error, result.error_type
                )
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.output})

    return traj


def _parse_final_answer(raw_arguments: str) -> str | None:
    try:
        args = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict) or "answer" not in args:
        return None
    return str(args["answer"])


def _tool_call_message(tc: ToolCall) -> dict:
    message = {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.name, "arguments": tc.arguments},
    }
    if tc.extra_content:
        message["extra_content"] = tc.extra_content
    return message
