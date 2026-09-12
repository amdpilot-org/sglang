"""Regression coverage for late prefill allocation misses (issue #34676)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=6, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_mod
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation import KVCacheOOMError, alloc_for_extend
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class _Req:
    def __init__(self, req_pool_idx=None):
        self.prefix_indices = torch.empty(0, dtype=torch.int64)
        self.dllm_incomplete_ids = None
        self.req_pool_idx = req_pool_idx
        self.kv = SimpleNamespace(holds_kv=False)


class _Allocator:
    page_size = 16

    def available_size(self):
        return 1_000_000

    def alloc_extend(self, *args, **kwargs):
        return None


class _TreeCache:
    page_size = 16

    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator

    def is_chunk_cache(self):
        return False

    def evict(self, params):
        pass

    def available_and_evictable_str(self):
        return "injected allocation miss"

    def pretty_print(self):
        pass


class _ReqPool:
    device = "cpu"

    def __init__(self):
        self.req_to_token = torch.zeros((8, 64), dtype=torch.int64)
        self.freed = []

    def alloc(self, reqs):
        for req in reqs:
            if req.req_pool_idx is None:
                req.req_pool_idx = 0
        return [req.req_pool_idx for req in reqs]

    def free(self, req):
        self.freed.append(req)
        req.req_pool_idx = None


class _Batch:
    def __init__(self, req):
        allocator = _Allocator()
        self.reqs = [req]
        self.prefix_lens = [0]
        self.extend_lens = [4]
        self.extend_num_tokens = 4
        self.device = "cpu"
        self.seq_lens = torch.tensor([4], dtype=torch.int64)
        self.seq_lens_cpu = torch.tensor([4], dtype=torch.int64)
        self.tree_cache = _TreeCache(allocator)
        self.req_to_token_pool = _ReqPool()

    def is_dllm(self):
        return False

    def maybe_evict_swa(self):
        pass


class TestAllocForExtendOOM(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", attention_backend="torch_native")
        )

    def test_real_paged_miss_is_typed_and_frees_new_req_slot(self):
        req = _Req()
        batch = _Batch(req)

        with self.assertRaises(KVCacheOOMError):
            alloc_for_extend(batch)

        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(batch.req_to_token_pool.freed, [req])

    def test_preexisting_req_slot_is_preserved(self):
        req = _Req(req_pool_idx=7)
        batch = _Batch(req)

        with self.assertRaises(KVCacheOOMError):
            alloc_for_extend(batch)

        self.assertEqual(req.req_pool_idx, 7)
        self.assertEqual(batch.req_to_token_pool.freed, [])


def _scheduler(waiting_req):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.grammar_manager = MagicMock(
        has_waiting_grammars=MagicMock(return_value=False)
    )
    scheduler.enable_hierarchical_cache = False
    scheduler.enable_unified_cache_external_linker = False
    scheduler.enable_hicache_storage = False
    scheduler.enable_priority_preemption = False
    scheduler.is_hybrid_swa = False
    scheduler.chunked_req = None
    scheduler.waiting_queue = [waiting_req]
    scheduler.min_free_slots_delayer = None
    scheduler.get_num_allocatable_reqs = MagicMock(return_value=64)
    scheduler.policy = MagicMock()
    scheduler.processed_tokens_counter = 0
    scheduler.chunked_prefill_size = 8192
    scheduler.dynamic_chunk_sizer = None
    scheduler.tp_worker = MagicMock()
    scheduler.page_size = 16
    scheduler.tree_cache = MagicMock()
    scheduler.token_to_kv_pool_allocator = MagicMock()
    scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
    scheduler.max_prefill_tokens = 8192
    scheduler.is_mixed_chunk = False
    scheduler.priority_scheduling_preemption_threshold = 0
    scheduler.max_prefill_bs = 64
    scheduler.max_running_requests = 64
    scheduler.dllm_config = None
    scheduler.enable_lora = False
    scheduler.req_to_token_pool = SimpleNamespace(available_size=lambda: 64)
    scheduler.disaggregation_mode = scheduler_mod.DisaggregationMode.NULL
    scheduler.truncation_align_size = None
    scheduler.enable_priority_scheduling = False
    scheduler.load_inquirer = MagicMock()
    scheduler.model_config = MagicMock()
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = MagicMock()
    scheduler._add_request_to_queue = lambda req: scheduler.waiting_queue.append(req)
    return scheduler


class TestSchedulerPrefillOOM(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_late_allocation_miss_requeues_instead_of_escaping(self):
        req = MagicMock(name="admitted_req")
        scheduler = _scheduler(req)
        adder = MagicMock(
            can_run_list=[req], preempt_list=[], new_chunked_req=None
        )
        adder.add_one_req.return_value = AddReqResult.CONTINUE
        batch = MagicMock()
        batch.prepare_for_extend.side_effect = KVCacheOOMError("injected")
        batch_cls = MagicMock()
        batch_cls.init_new.return_value = batch
        running_batch = MagicMock(reqs=[], batch_is_full=False)
        running_batch.is_empty.return_value = True

        with patch.object(scheduler_mod, "PrefillAdder", return_value=adder), patch.object(
            scheduler_mod, "ScheduleBatch", batch_cls
        ):
            result, returned_running = Scheduler._get_new_batch_prefill_raw(
                scheduler, None, running_batch
            )

        batch.prepare_for_extend.assert_called_once()
        self.assertIsNone(result)
        self.assertIs(returned_running, running_batch)
        self.assertIn(req, scheduler.waiting_queue)
        self.assertFalse(running_batch.batch_is_full)

    def test_empty_running_batch_retries_requeued_request(self):
        req = MagicMock(name="admitted_req")
        scheduler = _scheduler(req)
        adder = MagicMock(
            can_run_list=[req], preempt_list=[], new_chunked_req=None
        )
        adder.add_one_req.return_value = AddReqResult.CONTINUE
        batch = MagicMock()
        batch.prepare_for_extend.side_effect = KVCacheOOMError("injected")
        batch_cls = MagicMock()
        batch_cls.init_new.return_value = batch
        running_batch = MagicMock(reqs=[], batch_is_full=False)
        running_batch.is_empty.return_value = True

        with patch.object(scheduler_mod, "PrefillAdder", return_value=adder), patch.object(
            scheduler_mod, "ScheduleBatch", batch_cls
        ):
            Scheduler._get_new_batch_prefill_raw(scheduler, None, running_batch)
            Scheduler._get_new_batch_prefill_raw(scheduler, None, running_batch)

        self.assertEqual(batch.prepare_for_extend.call_count, 2)
        self.assertEqual(scheduler.waiting_queue, [req])

    def test_nonempty_running_batch_keeps_allocation_backpressure(self):
        req = MagicMock(name="admitted_req")
        scheduler = _scheduler(req)
        adder = MagicMock(
            can_run_list=[req], preempt_list=[], new_chunked_req=None
        )
        adder.add_one_req.return_value = AddReqResult.CONTINUE
        batch = MagicMock()
        batch.prepare_for_extend.side_effect = KVCacheOOMError("injected")
        batch_cls = MagicMock()
        batch_cls.init_new.return_value = batch
        running_batch = MagicMock(reqs=[MagicMock(name="decode_req")])
        running_batch.is_empty.return_value = False

        with patch.object(scheduler_mod, "PrefillAdder", return_value=adder), patch.object(
            scheduler_mod, "ScheduleBatch", batch_cls
        ):
            Scheduler._get_new_batch_prefill_raw(scheduler, None, running_batch)

        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(scheduler.waiting_queue, [req])

    def test_existing_chunked_req_is_not_duplicated_in_waiting_queue(self):
        req = MagicMock(name="active_chunked_req")
        req.inflight_middle_chunks = 0
        scheduler = _scheduler(req)
        scheduler.waiting_queue = []
        scheduler.chunked_req = req
        adder = MagicMock(can_run_list=[req], preempt_list=[], new_chunked_req=None)
        adder.add_chunked_req.return_value = req
        batch = MagicMock()
        batch.prepare_for_extend.side_effect = KVCacheOOMError("injected")
        batch_cls = MagicMock()
        batch_cls.init_new.return_value = batch
        running_batch = MagicMock(reqs=[], batch_is_full=False)
        running_batch.is_empty.return_value = True

        with patch.object(scheduler_mod, "PrefillAdder", return_value=adder), patch.object(
            scheduler_mod, "ScheduleBatch", batch_cls
        ):
            result, returned_running = Scheduler._get_new_batch_prefill_raw(
                scheduler, None, running_batch
            )

        self.assertIsNone(result)
        self.assertIs(returned_running, running_batch)
        self.assertIs(scheduler.chunked_req, req)
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertNotIn(req, scheduler.waiting_queue)
        self.assertFalse(running_batch.batch_is_full)

    def test_existing_chunked_req_does_not_block_other_requeues(self):
        chunked_req = MagicMock(name="active_chunked_req")
        chunked_req.inflight_middle_chunks = 0
        waiting_req = MagicMock(name="admitted_waiting_req")
        scheduler = _scheduler(waiting_req)
        scheduler.chunked_req = chunked_req
        adder = MagicMock(
            can_run_list=[chunked_req, waiting_req],
            preempt_list=[],
            new_chunked_req=None,
        )
        adder.add_chunked_req.return_value = chunked_req
        adder.add_one_req.return_value = AddReqResult.CONTINUE
        batch = MagicMock()
        batch.prepare_for_extend.side_effect = KVCacheOOMError("injected")
        batch_cls = MagicMock()
        batch_cls.init_new.return_value = batch
        running_batch = MagicMock(reqs=[], batch_is_full=False)

        with patch.object(scheduler_mod, "PrefillAdder", return_value=adder), patch.object(
            scheduler_mod, "ScheduleBatch", batch_cls
        ):
            Scheduler._get_new_batch_prefill_raw(scheduler, None, running_batch)

        self.assertIs(scheduler.chunked_req, chunked_req)
        self.assertEqual(chunked_req.inflight_middle_chunks, 0)
        self.assertEqual(scheduler.waiting_queue, [waiting_req])

    def test_new_chunked_req_is_cleared_and_requeued(self):
        req = MagicMock(name="new_chunked_req")
        req.inflight_middle_chunks = 0
        scheduler = _scheduler(req)
        adder = MagicMock(can_run_list=[req], preempt_list=[], new_chunked_req=req)
        adder.add_one_req.return_value = AddReqResult.CONTINUE
        batch = MagicMock()
        batch.prepare_for_extend.side_effect = KVCacheOOMError("injected")
        batch_cls = MagicMock()
        batch_cls.init_new.return_value = batch
        running_batch = MagicMock(reqs=[], batch_is_full=False)

        with patch.object(scheduler_mod, "PrefillAdder", return_value=adder), patch.object(
            scheduler_mod, "ScheduleBatch", batch_cls
        ):
            Scheduler._get_new_batch_prefill_raw(scheduler, None, running_batch)

        self.assertIsNone(scheduler.chunked_req)
        self.assertEqual(req.inflight_middle_chunks, 0)
        self.assertEqual(scheduler.waiting_queue, [req])


if __name__ == "__main__":
    import unittest

    unittest.main()
