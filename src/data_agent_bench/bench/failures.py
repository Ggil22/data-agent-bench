"""Failure taxonomy: assign each episode one primary outcome plus behaviour flags.

Primary outcomes are mutually exclusive and checked in priority order. Flags are
multi-label and describe behaviours seen during the episode, even in successes
(e.g. a model can recover from a hallucinated tool and still answer correctly).

These rules are heuristics over the trajectory. Planning or reasoning errors hide
inside `wrong_answer` and need qualitative review of the trajectories.
"""

from __future__ import annotations

from collections import Counter

from data_agent_bench.agent import Trajectory
from data_agent_bench.bench.scoring import Score

OUTCOMES = (
    "correct",
    "api_error",
    "loop",
    "max_steps",
    "hallucinated_answer",
    "false_refusal",
    "format_error",
    "answered_without_evidence",
    "wrong_answer",
)

FLAGS = (
    "tool_hallucination",
    "code_error",
    "repeated_call",
    "no_evidence",
    "text_only_turn",
)

_EVIDENCE_TOOLS = {"run_python", "inspect_csv"}


def behaviour_flags(traj: Trajectory) -> list[str]:
    exchanges = traj.exchanges
    flags = []
    if any(ex.error_type in ("unknown_tool", "invalid_arguments") for ex in exchanges):
        flags.append("tool_hallucination")
    if any(ex.error_type in ("execution_error", "timeout") for ex in exchanges):
        flags.append("code_error")
    calls = Counter((ex.name, ex.arguments) for ex in exchanges if ex.name != "final_answer")
    if calls and max(calls.values()) >= 3:
        flags.append("repeated_call")
    if not any(ex.name in _EVIDENCE_TOOLS and not ex.is_error for ex in exchanges):
        flags.append("no_evidence")
    if any(not step.tool_calls for step in traj.steps):
        flags.append("text_only_turn")
    return flags


def classify(traj: Trajectory, sc: Score, unanswerable_task: bool) -> tuple[str, list[str]]:
    flags = behaviour_flags(traj)
    if sc.correct:
        outcome = "correct"
    elif traj.stop_reason == "api_error":
        outcome = "api_error"
    elif traj.final_answer is None:
        outcome = "loop" if "repeated_call" in flags else "max_steps"
    elif unanswerable_task:
        outcome = "hallucinated_answer"
    elif sc.predicted_unanswerable:
        outcome = "false_refusal"
    elif sc.parse_error:
        outcome = "format_error"
    elif "no_evidence" in flags:
        outcome = "answered_without_evidence"
    else:
        outcome = "wrong_answer"
    return outcome, flags
