import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _WaitingQueue:
    def __init__(self):
        self.iteration_count = 0

    def __len__(self):
        return 1

    def __iter__(self):
        self.iteration_count += 1
        return iter(())


class _FakeAdder:
    budget_after_chunk = AddReqResult.OTHER

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None

    def add_chunked_req(self, req):
        self.can_run_list.append(req)
        return None

    def budget_state(self):
        return self.budget_after_chunk


def _scheduler(waiting_queue):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.grammar_manager = MagicMock(has_waiting_grammars=lambda: False)
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_unified_cache_external_linker = False
    scheduler.enable_hicache_storage = False
    scheduler.enable_priority_preemption = False
    scheduler.is_hybrid_swa = False
    scheduler.min_free_slots_delayer = None
    scheduler.get_num_allocatable_reqs = MagicMock(return_value=8)
    scheduler.policy = MagicMock()
    scheduler.processed_tokens_counter = 0
    scheduler.chunked_prefill_size = 4
    scheduler.dynamic_chunk_sizer = None
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(), prefill_aware_swa=False
        )
    )
    scheduler.page_size = 1
    scheduler.tree_cache = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.model_config = MagicMock()
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    scheduler.max_prefill_tokens = 32
    scheduler.is_mixed_chunk = False
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.max_prefill_bs = 8
    scheduler.max_running_requests = 8
    scheduler.dllm_config = None
    scheduler.enable_lora = False
    scheduler.req_to_token_pool = SimpleNamespace(mamba_allocator=None)
    scheduler.disaggregation_mode = None
    scheduler.truncation_align_size = None
    scheduler.spec_algorithm = MagicMock()
    scheduler.enable_overlap = False
    scheduler.enable_priority_scheduling = False
    scheduler.load_inquirer = MagicMock()
    scheduler.waiting_queue = waiting_queue
    scheduler.chunked_req = MagicMock()
    return scheduler


class TestSchedulerChunkBudgetGate(unittest.TestCase):
    def _run(self, budget_state):
        waiting_queue = _WaitingQueue()
        scheduler = _scheduler(waiting_queue)
        running_batch = MagicMock()
        running_batch.reqs = []
        running_batch.batch_is_full = False
        running_batch.is_empty.return_value = True
        _FakeAdder.budget_after_chunk = budget_state

        with patch("sglang.srt.managers.scheduler.PrefillAdder", _FakeAdder), patch(
            "sglang.srt.managers.scheduler.ScheduleBatch.init_new",
            return_value=MagicMock(),
        ), patch(
            "sglang.srt.managers.scheduler.get_memory",
            return_value=SimpleNamespace(enable_flexkv=False),
        ), patch(
            "sglang.srt.managers.scheduler.get_schedule",
            return_value=SimpleNamespace(prefill_max_requests=None),
        ), patch(
            "sglang.srt.managers.scheduler.PrefillStats.from_adder",
            return_value=MagicMock(),
        ), patch("sglang.srt.managers.scheduler.set_time_batch"):
            scheduler._get_new_batch_prefill_raw(None, running_batch)

        return waiting_queue.iteration_count

    def test_exhausted_chunk_budget_skips_waiting_queue(self):
        # The one iteration is the final queue-maintenance pass; the admission
        # scan itself must be skipped.
        self.assertEqual(self._run(AddReqResult.OTHER), 1)

    def test_remaining_chunk_budget_scans_waiting_queue(self):
        self.assertEqual(self._run(AddReqResult.CONTINUE), 2)

    def test_exhausted_total_budget_skips_waiting_queue(self):
        self.assertEqual(self._run(AddReqResult.NO_TOKEN), 1)


if __name__ == "__main__":
    unittest.main()
