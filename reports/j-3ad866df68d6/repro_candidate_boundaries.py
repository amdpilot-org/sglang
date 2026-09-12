import importlib
import multiprocessing
import os
import sys
import tempfile
import time
import unittest.mock

GIB = 1024**3


def load_base():
    sys.path.insert(0, "python")
    return importlib.import_module("sglang.srt.mem_cache.pool_host.base")


def reset(base):
    base._initial_host_memory_available_bytes = None
    base._reserved_host_memory_bytes = 0


def unequal_in_job(base):
    reset(base)
    available = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 90 * GIB
    with (
        unittest.mock.patch.object(base, "ranks_per_host", return_value=4),
        unittest.mock.patch.object(base, "host_memory_sync_group", return_value=None),
        unittest.mock.patch.object(
            base.psutil,
            "virtual_memory",
            return_value=unittest.mock.Mock(available=available),
        ),
    ):
        budget = base.host_memory_budget_bytes(30 * GIB)
    print(
        "UNEQUAL_IN_JOB",
        {"request_gib": 30, "budget_gib": budget / GIB, "aggregate_gib": 90},
    )
    assert budget >= 30 * GIB, "aggregate-fitting 30 GiB rank was rejected"


def separate_jobs(base):
    outcomes = []
    available = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 90 * GIB
    for _job in range(2):
        reset(base)
        with (
            unittest.mock.patch.object(base, "ranks_per_host", return_value=1),
            unittest.mock.patch.object(
                base, "host_memory_sync_group", return_value=None
            ),
            unittest.mock.patch.object(
                base.psutil,
                "virtual_memory",
                return_value=unittest.mock.Mock(available=available),
            ),
        ):
            outcomes.append(base.host_memory_budget_bytes(60 * GIB) >= 60 * GIB)
    print(
        "SEPARATE_CONCURRENT_JOBS",
        {"accepted": outcomes, "aggregate_gib": 120, "usable_gib": 90},
    )
    assert not all(outcomes), (
        "independent jobs both accepted an oversubscribed aggregate"
    )


def corrected_worker(request_gib, available_gib, results, lock_path):
    base = load_base()
    base._HOST_MEMORY_ALLOCATION_LOCK_PATH = lock_path
    with unittest.mock.patch.object(
        base.psutil,
        "virtual_memory",
        side_effect=lambda: unittest.mock.Mock(
            available=base.HICACHE_HOST_MEMORY_RESERVE_BYTES + available_gib.value * GIB
        ),
    ):
        with base.host_memory_allocation_lock() as budget:
            accepted = request_gib * GIB <= budget
            if accepted:
                time.sleep(0.05)
                available_gib.value -= request_gib
    results.put((request_gib, accepted))


def corrected_process_case(requests):
    ctx = multiprocessing.get_context("spawn")
    available_gib = ctx.Value("q", 90, lock=False)
    results = ctx.Queue()
    lock_path = os.path.join(tempfile.mkdtemp(), "allocation.lock")
    processes = [
        ctx.Process(
            target=corrected_worker,
            args=(request, available_gib, results, lock_path),
        )
        for request in requests
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0
    outcomes = sorted(results.get() for _ in processes)
    print(
        "CORRECTED_PROCESSES",
        {
            "requests_gib": requests,
            "outcomes": outcomes,
            "remaining_gib": available_gib.value,
        },
    )
    return outcomes


if __name__ == "__main__":
    base = load_base()
    if hasattr(base, "host_memory_allocation_lock"):
        unequal = corrected_process_case([30, 20, 20, 20])
        separate = corrected_process_case([60, 60])
        assert all(accepted for _, accepted in unequal)
        assert sum(accepted for _, accepted in separate) == 1
        raise SystemExit(0)
    failures = 0
    for check in (unequal_in_job, separate_jobs):
        try:
            check(base)
        except AssertionError as exc:
            failures += 1
            print(type(exc).__name__ + ":", exc)
    raise SystemExit(failures)
