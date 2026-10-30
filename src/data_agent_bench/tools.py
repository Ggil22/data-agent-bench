"""Tools available to the data analyst agent.

The same implementations back both the benchmark agent loop (via `ToolExecutor.execute`,
which turns every failure into a typed `ToolResult`) and the MCP server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from data_agent_bench.sandbox import Sandbox, SubprocessSandbox

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files available in the working directory.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_csv",
            "description": (
                "Show the shape, column types, missing value counts and first rows of a CSV file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "CSV file name, e.g. sales.csv"},
                    "n_rows": {"type": "integer", "description": "Rows to preview (1-20)."},
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code in the working directory. pandas and numpy are available. "
                "Each call is a fresh process (no state is kept between calls). Use print() to "
                "show results; the value of the last expression is printed too."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source code."}},
                "required": ["code"],
            },
        },
    },
]

TOOL_NAMES = frozenset(spec["function"]["name"] for spec in TOOL_SPECS)


class ToolError(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class ToolResult:
    output: str
    is_error: bool = False
    # unknown_tool | invalid_arguments | file_not_found | execution_error | timeout
    error_type: str | None = None


class ToolExecutor:
    def __init__(
        self,
        workdir: Path,
        sandbox: Sandbox | None = None,
        timeout_s: float = 30.0,
        max_output_chars: int = 4000,
    ):
        self.workdir = Path(workdir).resolve()
        self.sandbox = sandbox or SubprocessSandbox()
        self.timeout_s = timeout_s
        self.max_output_chars = max_output_chars

    def list_files(self) -> str:
        files = sorted(p for p in self.workdir.iterdir() if p.is_file())
        if not files:
            return "(no files)"
        return "\n".join(f"{p.name}\t{p.stat().st_size} bytes" for p in files)

    def inspect_csv(self, filename: str, n_rows: int = 5) -> str:
        path = self._resolve(filename)
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ToolError("execution_error", f"Could not read {filename} as CSV: {exc}") from exc
        n_rows = max(1, min(int(n_rows), 20))
        schema = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non_null": df.notna().sum(),
                "missing": df.isna().sum(),
            }
        )
        text = (
            f"{filename}: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
            f"{schema.to_string()}\n\nFirst {n_rows} rows:\n{df.head(n_rows).to_string()}"
        )
        return self._truncate(text)

    def run_python(self, code: str) -> str:
        result = self.sandbox.run(code, self.workdir, self.timeout_s)
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + "[stderr]\n" + result.stderr
        output = self._truncate(output.strip())
        if result.timed_out:
            raise ToolError(
                "timeout", f"Execution timed out after {self.timeout_s:.0f}s.\n{output}"
            )
        if result.returncode != 0:
            raise ToolError("execution_error", output or "Process exited with an error.")
        return output or "(no output; use print() to display results)"

    def execute(self, name: str, raw_arguments: str) -> ToolResult:
        if name not in TOOL_NAMES:
            available = ", ".join(sorted(TOOL_NAMES | {"final_answer"}))
            return ToolResult(
                f"Error: unknown tool '{name}'. Available tools: {available}.", True, "unknown_tool"
            )
        try:
            args = json.loads(raw_arguments) if raw_arguments.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            output = getattr(self, name)(**args)
        except (ValueError, TypeError) as exc:
            return ToolResult(
                f"Error: invalid arguments for {name}: {exc}", True, "invalid_arguments"
            )
        except ToolError as exc:
            return ToolResult(f"Error: {exc}", True, exc.error_type)
        return ToolResult(output)

    def _resolve(self, filename: str) -> Path:
        path = (self.workdir / filename).resolve()
        if path.parent != self.workdir:
            raise ToolError("invalid_arguments", "Only files in the working directory are allowed.")
        if not path.is_file():
            raise ToolError("file_not_found", f"File '{filename}' does not exist.")
        return path

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        omitted = len(text) - self.max_output_chars
        return text[: self.max_output_chars] + f"\n... [truncated {omitted} characters]"
