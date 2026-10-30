"""MCP server exposing the data analyst tools to any MCP client (Claude Desktop, IDEs...).

Usage: dab-mcp [DATA_DIR]   (defaults to $DAB_DATA_DIR or ./data)

Note: tools operate directly on DATA_DIR, so code run through `run_python` can
modify files there. Point it at a copy of your data.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from data_agent_bench.tools import ToolExecutor


def build_server(data_dir: Path) -> FastMCP:
    executor = ToolExecutor(data_dir)
    server = FastMCP("data-analyst")

    def _call(name: str, **kwargs) -> str:
        return executor.execute(name, json.dumps(kwargs)).output

    @server.tool()
    def list_files() -> str:
        """List the files available in the data directory."""
        return _call("list_files")

    @server.tool()
    def inspect_csv(filename: str, n_rows: int = 5) -> str:
        """Show the shape, column types, missing value counts and first rows of a CSV file."""
        return _call("inspect_csv", filename=filename, n_rows=n_rows)

    @server.tool()
    def run_python(code: str) -> str:
        """Run Python code (pandas, numpy available) in the data directory. Each call is a fresh
        process; use print() to show results. The value of the last expression is printed too."""
        return _call("run_python", code=code)

    return server


def main() -> None:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DAB_DATA_DIR", "data"))
    if not data_dir.is_dir():
        sys.exit(f"Data directory not found: {data_dir}")
    build_server(data_dir.resolve()).run()


if __name__ == "__main__":
    main()
