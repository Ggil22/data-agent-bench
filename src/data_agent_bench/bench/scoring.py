"""Answer parsing and scoring against ground truth."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from data_agent_bench.bench.tasks import Task

UNANSWERABLE = "UNANSWERABLE"

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


@dataclass
class Score:
    correct: bool
    parsed: Any = None
    parse_error: bool = False
    predicted_unanswerable: bool = False


def normalize(text: str) -> str:
    return " ".join(text.strip().strip("\"'`.").lower().split())


def parse_number(text: str) -> float | None:
    cleaned = _THOUSANDS.sub("", text.replace("−", "-").replace("%", ""))
    match = _NUMBER.search(cleaned)
    return float(match.group()) if match else None


def numbers_match(pred: float, truth: float, decimals: int | None) -> bool:
    # A correctly rounded answer is within half a unit of the last requested decimal.
    tolerance = 0.5 * 10 ** -(decimals or 0) + 1e-9
    return math.isclose(pred, truth, rel_tol=1e-9, abs_tol=tolerance)


def score(task: Task, truth: Any, prediction: str | None) -> Score:
    if prediction is None:
        return Score(correct=False)
    predicted_unanswerable = UNANSWERABLE.lower() in normalize(prediction)
    if task.answer_type == "unanswerable":
        return Score(predicted_unanswerable, prediction, False, predicted_unanswerable)
    if predicted_unanswerable:
        return Score(False, prediction, False, True)

    if task.answer_type == "number":
        value = parse_number(prediction)
        if value is None:
            return Score(False, None, parse_error=True)
        return Score(numbers_match(value, float(truth), task.decimals), value)

    if task.answer_type == "string":
        value = normalize(prediction)
        return Score(value == normalize(str(truth)), value)

    if task.answer_type == "bool":
        words = normalize(prediction).split()
        first = words[0].strip(".,!") if words else ""
        value = {"yes": True, "true": True, "no": False, "false": False}.get(first)
        if value is None:
            return Score(False, None, parse_error=True)
        return Score(value == truth, value)

    if task.answer_type == "list_ordered":
        items = [normalize(i) for i in re.split(r"[,;\n]", prediction) if i.strip()]
        expected = [normalize(str(t)) for t in truth]
        return Score(len(items) == len(expected) and all(map(_items_match, items, expected)), items)

    raise ValueError(f"Unknown answer type: {task.answer_type}")


def _items_match(pred: str, truth: str) -> bool:
    p, t = parse_number(pred), parse_number(truth)
    if p is not None and t is not None and _NUMBER.fullmatch(pred) and _NUMBER.fullmatch(truth):
        return math.isclose(p, t)
    return pred == truth
