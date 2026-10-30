import pytest

from data_agent_bench.bench.scoring import numbers_match, parse_number, score
from data_agent_bench.bench.tasks import Task


def make_task(answer_type, decimals=None):
    return Task("t", "c", "easy", "q", answer_type, lambda d: None, decimals)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("42", 42.0),
        ("1,234.56", 1234.56),
        ("-3.45%", -3.45),
        ("The answer is 7.", 7.0),
        ("−2.5", -2.5),
        ("abc", None),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_number_tolerance_follows_requested_rounding():
    assert numbers_match(12.35, 12.3456, decimals=2)
    assert not numbers_match(12.36, 12.3456, decimals=2)
    assert numbers_match(2015, 2015.0, decimals=None)
    assert not numbers_match(2016, 2015.0, decimals=None)


def test_number_scoring():
    assert score(make_task("number", 2), 1234.567, "1,234.57").correct
    assert score(make_task("number", 2), 1234.567, "abc").parse_error


def test_string_scoring_is_normalized():
    assert score(make_task("string"), "East", " east.").correct
    assert not score(make_task("string"), "East", "West").correct


def test_bool_scoring():
    assert score(make_task("bool"), True, "Yes").correct
    assert score(make_task("bool"), False, "no.").correct
    assert score(make_task("bool"), True, "maybe").parse_error


def test_list_scoring_is_ordered():
    task = make_task("list_ordered")
    assert score(task, [12, 7, 3], "12, 7, 3").correct
    assert not score(task, [12, 7, 3], "7, 12, 3").correct
    assert not score(task, [12, 7, 3], "12, 7").correct


def test_unanswerable():
    task = make_task("unanswerable")
    assert score(task, None, "UNANSWERABLE").correct
    assert not score(task, None, "34.5").correct
    refusal = score(make_task("number"), 10, "unanswerable")
    assert not refusal.correct and refusal.predicted_unanswerable


def test_missing_prediction():
    assert not score(make_task("number"), 1, None).correct
