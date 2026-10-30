# data-agent-bench

A tool-using **data analyst agent** and a **verifiable benchmark** to measure how well
different LLMs solve multi-step data analysis tasks: which models succeed, on what kind
of task, and *how* they fail.

The agent receives CSV files and a question, explores the data, writes and runs pandas
code in a sandbox, and submits an answer. Every answer is checked against ground truth
computed by code, and every failure is classified with a failure taxonomy.

> **Status:** v0.1. Infrastructure, 34 tasks and the evaluation harness are done.
> Benchmark results will be published here once the first full run is complete.

## Why

Agent demos are easy; knowing whether an agent *works* is not. This project focuses
on the evaluation side:

- **Verifiable tasks.** Synthetic datasets generated from a fixed seed, so ground truth
  is computed by code, never hand-labelled, and cannot have leaked into training data.
- **Realistic traps.** Exact duplicate rows, missing values, a `-999` sensor error code
  and gaps in a time series, documented only in `DATA_DICTIONARY.md`, as in real projects.
- **Unanswerable questions** to measure hallucination: the right answer is `UNANSWERABLE`.
- **Failure taxonomy**, not just accuracy (see below).
- **Honest statistics.** 95% Wilson confidence intervals; with 35 tasks, small differences
  between models are noise and the report makes that visible.

## Architecture

```
question + CSV files
        |
        v
  +-----------+   tool calls    +--------------------------------------+
  |   Agent   | --------------> | Tools                                |
  |   loop    | <-------------- |  list_files / inspect_csv /          |
  +-----------+   tool results  |  run_python (sandboxed subprocess)   |
        |                       +--------------------------------------+
        | final_answer                     ^
        v                                  | same tools exposed over MCP
  scoring + failure taxonomy        dab-mcp server (Claude Desktop, IDEs...)
        |
        v
  results.jsonl -> report.md / summary.csv
```

| Module | Role |
|---|---|
| `llm.py` | Provider-agnostic client for any OpenAI-compatible API, with disk cache, rate limiting and retries |
| `sandbox.py` | Runs agent code in an isolated process: timeout, minimal env, network and subprocess blocked |
| `tools.py` | Tool implementations and JSON schemas; every failure becomes a typed error |
| `agent.py` | The agent loop and trajectory recording |
| `mcp_server.py` | The same tools exposed as an MCP server |
| `bench/datasets.py` | Deterministic synthetic datasets |
| `bench/tasks.py` | 35 tasks across 9 categories; ground truth computed from the data |
| `bench/scoring.py` | Answer parsing and tolerance-aware scoring |
| `bench/failures.py` | Failure taxonomy |
| `bench/runner.py`, `bench/report.py` | Resumable benchmark runs and reports |

## Failure taxonomy

Each episode gets exactly one **outcome**:

| Outcome | Meaning |
|---|---|
| `correct` | Right answer |
| `loop` | Hit the step limit while repeating the same tool call |
| `max_steps` | Hit the step limit without a final answer |
| `hallucinated_answer` | Gave an answer to an unanswerable question |
| `false_refusal` | Said UNANSWERABLE to an answerable question |
| `format_error` | Answer could not be parsed (e.g. prose instead of a number) |
| `answered_without_evidence` | Answered without ever successfully running code or inspecting data |
| `wrong_answer` | Everything else: analysis, planning or calculation errors |
| `api_error` | Provider failure (excluded from model comparisons; retried on resume) |

Plus multi-label **behaviour flags**, recorded even on successes: `tool_hallucination`
(called a non-existent tool or sent invalid arguments), `code_error`, `repeated_call`,
`no_evidence`, `text_only_turn`.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env        # add at least one free API key (see below)
uv run dab models           # shows which models are ready
uv run dab tasks            # lists the benchmark tasks
uv run dab run --models gemini-flash-lite --tasks category:join   # quick smoke run
uv run dab run --models gemini-flash-lite groq-llama-70b --run-id main
```

Results go to `results/<run-id>/`: `results.jsonl` (one record per episode, including the
full trajectory), `report.md` and `summary.csv`. If a run is interrupted (free-tier rate
limits), run the same command with the same `--run-id` to resume.

### Free model providers

The whole project runs at zero cost. All providers in `configs/models.yaml` offer an
OpenAI-compatible API with a free tier: Google AI Studio (Gemini), Groq, OpenRouter
(`:free` models), Cerebras, and local models through Ollama. Free-tier limits and model
ids change often: check them before a run and adjust `rpm` in the config.

Things learned the hard way with Gemini (October 2026):

- **Retired models still show up in the model list.** `gemini-2.5-flash` is listed by
  `/models` but returns `404` on chat requests; the config now uses `gemini-3.5-flash`
  and `gemini-3.5-flash-lite`.
- **Daily quotas are the real bottleneck, not requests per minute.** On the free tier
  `gemini-3.5-flash` allows only 20 requests per day per model. An episode takes ~5
  requests, so that is about 4 tasks a day. `gemini-3.5-flash-lite` has a much larger
  quota and completes the full benchmark in one go.

LLM responses are cached in `.cache/llm/`, so re-running or re-scoring is free.

### MCP server

```bash
uv run dab data             # writes the datasets to ./data
uv run dab-mcp ./data       # stdio MCP server with list_files, inspect_csv, run_python
```

## Design decisions

- **Stateless code execution.** Each `run_python` call is a fresh process. This costs some
  re-loading, but makes episodes reproducible and isolates failures. A persistent kernel
  is a possible ablation.
- **Notebook-style output.** The value of a trailing expression is printed, because models
  frequently write `df.head()` expecting to see it. Without this, many failures are just
  missing `print()` calls.
- **Provider-specific tool-call fields are echoed back.** Gemini 3 attaches a
  `thought_signature` (in `extra_content`) to each tool call and rejects the next turn
  with `400` if it is missing from the history. `ToolCall.extra_content` keeps it and the
  agent sends it back unchanged; other providers simply leave it empty.
- **Answers via a `final_answer` tool**, not free text, so answers can be parsed reliably
  and format adherence can be measured on its own.
- **Tolerance follows the requested rounding**: an answer is correct if it is within half a
  unit of the last requested decimal.
- **Sandbox limits.** The subprocess sandbox blocks network and process creation with
  Python audit hooks. This is best-effort isolation, fine for cooperative models on
  synthetic data. Running untrusted code needs a container or VM sandbox (the `Sandbox`
  protocol makes that a drop-in replacement).

## Development

```bash
uv run pytest           # tests use a scripted fake LLM: no API keys needed
uv run ruff check .
```

## Roadmap

- [ ] First full benchmark run across 4-6 free models
- [ ] Grow to 60-100 tasks, including hand-written hard cases
- [ ] Qualitative error analysis of `wrong_answer` trajectories
- [ ] Ablations: persistent kernel vs stateless, with vs without data dictionary hint
- [ ] Docker sandbox backend
