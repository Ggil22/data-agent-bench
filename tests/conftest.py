import json
from pathlib import Path

import pytest

from data_agent_bench.bench import datasets
from data_agent_bench.llm import LLMResponse, ToolCall


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    return datasets.write(tmp_path_factory.mktemp("data"))


@pytest.fixture(scope="session")
def data(data_dir):
    return datasets.load(data_dir)


class ScriptedLLM:
    """Replays a fixed list of responses; stands in for a real model in tests."""

    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    def chat(self, messages, tools):
        self.seen_messages.append(list(messages))
        if not self.responses:
            return LLMResponse(content="I am not sure.")
        return self.responses.pop(0)


def call(name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        content=None, tool_calls=[ToolCall(f"call_{name}", name, json.dumps(arguments))]
    )
