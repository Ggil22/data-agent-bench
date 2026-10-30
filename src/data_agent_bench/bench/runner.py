"""Run models over benchmark tasks and store one JSON record per episode.

Runs are resumable: episodes already in results.jsonl are skipped (except API
errors, which are retried), so a run interrupted by free-tier rate limits can be
continued with the same command.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from data_agent_bench.agent import run_agent
from data_agent_bench.bench import datasets
from data_agent_bench.bench.failures import classify
from data_agent_bench.bench.scoring import score
from data_agent_bench.bench.tasks import Task
from data_agent_bench.llm import LLMClient
from data_agent_bench.sandbox import SubprocessSandbox
from data_agent_bench.tools import ToolExecutor

ClientFactory = Callable[[str, int], LLMClient]  # (model name, repeat index) -> client


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def latest_records(records: list[dict]) -> list[dict]:
    """Keep the last record per (model, task, repeat), so retried episodes replace old ones."""
    latest = {(r["model"], r["task_id"], r["repeat"]): r for r in records}
    return list(latest.values())


def run_benchmark(
    model_names: list[str],
    tasks: list[Task],
    make_client: ClientFactory,
    data_dir: Path,
    out_dir: Path,
    repeats: int = 1,
    max_steps: int = 12,
    log: Callable[[str], None] = print,
) -> Path:
    data = datasets.load(data_dir)
    truths = {t.id: t.compute(data) for t in tasks}
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = {
        (r["model"], r["task_id"], r["repeat"])
        for r in latest_records(load_records(results_path))
        if r["outcome"] != "api_error"
    }
    sandbox = SubprocessSandbox()

    for model in model_names:
        for repeat in range(repeats):
            client = make_client(model, repeat)
            for i, task in enumerate(tasks, 1):
                if (model, task.id, repeat) in done:
                    continue
                with tempfile.TemporaryDirectory(
                    prefix="dab_task_", ignore_cleanup_errors=True
                ) as tmp:
                    workdir = Path(tmp)
                    for f in data_dir.iterdir():
                        if f.is_file():
                            shutil.copy2(f, workdir / f.name)
                    executor = ToolExecutor(workdir, sandbox=sandbox)
                    traj = run_agent(client, task.question, executor, max_steps=max_steps)

                sc = score(task, truths[task.id], traj.final_answer)
                outcome, flags = classify(traj, sc, task.answer_type == "unanswerable")
                record = {
                    "model": model,
                    "task_id": task.id,
                    "category": task.category,
                    "difficulty": task.difficulty,
                    "repeat": repeat,
                    "question": task.question,
                    "truth": _jsonable(truths[task.id]),
                    "prediction": traj.final_answer,
                    "correct": sc.correct,
                    "outcome": outcome,
                    "flags": flags,
                    "stop_reason": traj.stop_reason,
                    "error": traj.error,
                    "n_steps": len(traj.steps),
                    "prompt_tokens": traj.prompt_tokens,
                    "completion_tokens": traj.completion_tokens,
                    "latency_s": round(traj.latency_s, 3),
                    "trajectory": [asdict(s) for s in traj.steps],
                }
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                mark = "OK  " if sc.correct else "FAIL"
                log(
                    f"[{model} r{repeat}] {i:>3}/{len(tasks)} {task.id:<9} {mark} {outcome:<26} "
                    f"pred={traj.final_answer!r:.40} truth={_jsonable(truths[task.id])!r:.40}"
                )
    return results_path


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value
