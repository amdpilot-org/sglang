from types import SimpleNamespace

from sglang.srt.managers.scheduler_components.load_inquirer import (
    SchedulerLoadInquirer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _batch(*rids: str):
    return SimpleNamespace(reqs=[SimpleNamespace(rid=rid) for rid in rids])


def test_num_running_reqs_counts_all_pipeline_slots():
    inquirer = object.__new__(SchedulerLoadInquirer)
    object.__setattr__(
        inquirer,
        "get_running_batches",
        lambda: [_batch("req-0"), _batch("req-1", "req-2"), _batch("req-3")],
    )

    assert inquirer._get_num_running_reqs() == 4


def test_num_running_reqs_deduplicates_requests_across_slots():
    inquirer = object.__new__(SchedulerLoadInquirer)
    object.__setattr__(
        inquirer,
        "get_running_batches",
        lambda: [_batch("req-0", "req-1"), _batch("req-1", "req-2")],
    )

    assert inquirer._get_num_running_reqs() == 3


def test_num_running_reqs_handles_single_and_empty_batches():
    inquirer = object.__new__(SchedulerLoadInquirer)
    object.__setattr__(inquirer, "get_running_batches", lambda: [_batch("req-0")])
    assert inquirer._get_num_running_reqs() == 1

    object.__setattr__(inquirer, "get_running_batches", lambda: [])
    assert inquirer._get_num_running_reqs() == 0
