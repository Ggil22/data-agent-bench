import shutil

import pytest

from data_agent_bench.tools import ToolExecutor


@pytest.fixture
def executor(data_dir, tmp_path):
    for f in data_dir.iterdir():
        shutil.copy2(f, tmp_path / f.name)
    return ToolExecutor(tmp_path)


def test_list_files(executor):
    result = executor.execute("list_files", "")
    assert not result.is_error
    assert "sales.csv" in result.output
    assert "DATA_DICTIONARY.md" in result.output


def test_inspect_csv(executor):
    result = executor.execute("inspect_csv", '{"filename": "sales.csv", "n_rows": 3}')
    assert not result.is_error
    assert "rows x 8 columns" in result.output
    assert "discount" in result.output


def test_unknown_tool(executor):
    result = executor.execute("query_database", "{}")
    assert result.error_type == "unknown_tool"


def test_invalid_arguments(executor):
    assert executor.execute("inspect_csv", "{not json").error_type == "invalid_arguments"
    assert (
        executor.execute("inspect_csv", '{"file": "sales.csv"}').error_type == "invalid_arguments"
    )


def test_path_traversal_is_rejected(executor):
    result = executor.execute("inspect_csv", '{"filename": "../secret.csv"}')
    assert result.error_type == "invalid_arguments"


def test_missing_file(executor):
    assert (
        executor.execute("inspect_csv", '{"filename": "nope.csv"}').error_type == "file_not_found"
    )


def test_run_python_error_is_typed(executor):
    result = executor.execute(
        "run_python", '{"code": "import pandas as pd\\npd.read_csv(\'x.csv\')"}'
    )
    assert result.error_type == "execution_error"
    assert "FileNotFoundError" in result.output


def test_output_is_truncated(executor):
    result = executor.execute("run_python", '{"code": "print(\'x\' * 10000)"}')
    assert "[truncated" in result.output
