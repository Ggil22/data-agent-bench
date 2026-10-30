from data_agent_bench.sandbox import SubprocessSandbox

sandbox = SubprocessSandbox()


def test_prints_stdout(tmp_path):
    result = sandbox.run("print('hello')", tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "hello"


def test_echoes_last_expression(tmp_path):
    result = sandbox.run("x = 20\nx + 22", tmp_path)
    assert result.stdout.strip() == "42"


def test_reports_exceptions(tmp_path):
    result = sandbox.run("1 / 0", tmp_path)
    assert result.returncode != 0
    assert "ZeroDivisionError" in result.stderr


def test_syntax_error(tmp_path):
    result = sandbox.run("def broken(:", tmp_path)
    assert result.returncode != 0
    assert "SyntaxError" in result.stderr


def test_timeout(tmp_path):
    result = sandbox.run("while True:\n    pass", tmp_path, timeout_s=2)
    assert result.timed_out


def test_blocks_network(tmp_path):
    result = sandbox.run("import socket\nsocket.create_connection(('example.com', 80))", tmp_path)
    assert result.returncode != 0
    assert "PermissionError" in result.stderr


def test_blocks_subprocess(tmp_path):
    result = sandbox.run("import subprocess\nsubprocess.run(['echo', 'hi'])", tmp_path)
    assert result.returncode != 0
    assert "PermissionError" in result.stderr


def test_runs_in_workdir_with_pandas(tmp_path):
    (tmp_path / "t.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    result = sandbox.run("import pandas as pd\nprint(pd.read_csv('t.csv')['a'].sum())", tmp_path)
    assert result.stdout.strip() == "4"


def test_unicode_output(tmp_path):
    result = sandbox.run("print('año 2024 — ok')", tmp_path)
    assert "año 2024 — ok" in result.stdout
