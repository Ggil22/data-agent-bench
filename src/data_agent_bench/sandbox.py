"""Execution sandbox for agent-written Python code.

`SubprocessSandbox` runs each snippet in a fresh, isolated Python process with a
timeout, a minimal environment and an audit hook that blocks network access and
process spawning. This is *best-effort* isolation: audit hooks can be bypassed by
determined code (e.g. via ctypes), and there are no memory limits. It is adequate
for benchmarking cooperative models on synthetic data; for untrusted workloads use
a container or VM based sandbox implementing the same `Sandbox` protocol.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

RUNNER_SOURCE = r"""
import ast
import sys
import traceback

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:  # preload heavy libraries before the audit hook is installed
    import numpy  # noqa: F401
    import pandas  # noqa: F401
except ImportError:
    pass

_BLOCKED = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.spawn",
    "os.posix_spawn", "os.startfile", "os.fork", "os.kill",
)


def _audit(event, args):
    if event.startswith(_BLOCKED):
        raise PermissionError(f"sandbox: '{event}' is not allowed")


with open(sys.argv[1], encoding="utf-8") as f:
    source = f.read()

sys.addaudithook(_audit)

try:
    tree = ast.parse(source, filename="<agent_code>")
except SyntaxError:
    traceback.print_exc(limit=0)
    sys.exit(1)

# Notebook-like behaviour: print the value of a trailing expression.
last_expr = None
if tree.body and isinstance(tree.body[-1], ast.Expr):
    last_expr = ast.Expression(tree.body.pop().value)

namespace = {"__name__": "__main__"}
try:
    exec(compile(tree, "<agent_code>", "exec"), namespace)
    if last_expr is not None:
        value = eval(compile(last_expr, "<agent_code>", "eval"), namespace)
        if value is not None:
            print(repr(value))
except BaseException as exc:
    traceback.print_exception(type(exc), exc, exc.__traceback__.tb_next)
    sys.exit(1)
"""


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    duration_s: float


class Sandbox(Protocol):
    def run(self, code: str, workdir: Path, timeout_s: float) -> ExecResult: ...


class SubprocessSandbox:
    def __init__(self, python: str | None = None):
        self.python = python or sys.executable
        self._scratch = Path(tempfile.mkdtemp(prefix="dab_sandbox_"))
        self._runner = self._scratch / "runner.py"
        self._runner.write_text(RUNNER_SOURCE, encoding="utf-8")

    def run(self, code: str, workdir: Path, timeout_s: float = 30.0) -> ExecResult:
        # Code lives outside the workdir so the agent never sees sandbox files.
        fd, code_path = tempfile.mkstemp(suffix=".py", dir=self._scratch)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [self.python, "-I", str(self._runner), code_path],
                cwd=workdir,
                env=self._env(workdir),
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecResult(
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr),
                returncode=-1,
                timed_out=True,
                duration_s=time.perf_counter() - start,
            )
        finally:
            Path(code_path).unlink(missing_ok=True)
        return ExecResult(
            stdout=_decode(proc.stdout),
            stderr=_decode(proc.stderr),
            returncode=proc.returncode,
            timed_out=False,
            duration_s=time.perf_counter() - start,
        )

    @staticmethod
    def _env(workdir: Path) -> dict[str, str]:
        env = {"TEMP": str(workdir), "TMP": str(workdir), "TMPDIR": str(workdir)}
        if "SYSTEMROOT" in os.environ:  # required by Python on Windows
            env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        return env


def _decode(data: bytes | None) -> str:
    return data.decode("utf-8", errors="replace") if data else ""
