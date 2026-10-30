import pandas as pd

from data_agent_bench.bench import datasets
from data_agent_bench.bench.tasks import ANSWER_TYPES, TASKS, select_tasks


def test_task_ids_are_unique():
    assert len({t.id for t in TASKS}) == len(TASKS)


def test_every_task_has_a_well_defined_answer(data):
    # compute() raises on ties, so this also checks no answer is ambiguous
    for task in TASKS:
        assert task.answer_type in ANSWER_TYPES, task.id
        truth = task.compute(data)
        if task.answer_type == "unanswerable":
            assert truth is None
        else:
            assert truth is not None, task.id


def test_generation_is_deterministic():
    a, b = datasets.generate(seed=7), datasets.generate(seed=7)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_data_contains_the_documented_traps(data):
    assert data["sales"].duplicated().sum() > 0
    assert data["sales"]["discount"].isna().sum() > 0
    assert (data["sensors"]["temperature"] == -999).sum() > 0


def test_select_tasks():
    assert len(select_tasks("all")) == len(TASKS)
    assert all(t.category == "join" for t in select_tasks("category:join"))
    assert [t.id for t in select_tasks("agg_001,un_001")] == ["agg_001", "un_001"]


def test_generate_writes_every_table(data):
    assert set(data) == set(datasets.TABLES)


def test_original_tables_do_not_depend_on_new_ones():
    # New tables use their own generators, so adding them must not change existing data
    # (otherwise results from earlier runs would no longer be comparable).
    tables = datasets.generate()
    sales = tables["sales"]
    assert len(sales) == 2015
    assert sales["units"].sum() == 11333


def test_new_traps_change_the_naive_answer(data):
    by_id = {t.id: t for t in TASKS}
    web, tickets = data["web_orders"], data["tickets"]
    # case and spacing variants inflate a naive distinct count
    assert web["customer_email"].nunique() > by_id["wo_003"].compute(data)
    # without stripping and normalising labels, many High tickets are missed
    assert (tickets["priority"] == "High").sum() < by_id["tk_001"].compute(data)
    # test orders dominate the naive top spender
    completed = web[web["status"] == "completed"]
    assert completed.groupby("customer_email")["amount"].sum().idxmax().endswith("@example.com")
    # Simpson's paradox: Tier2 is faster overall but slower on Medium tickets
    assert by_id["tk_005"].compute(data) == "Tier2"
    assert by_id["tk_007"].compute(data) == "Tier1"
