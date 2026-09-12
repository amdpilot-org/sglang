import asyncio
import unittest
from unittest.mock import Mock

from sglang.srt.utils.weight_versions import WeightVersionSpan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    ContinueGenerationReqInput,
    PauseContinueBroadcastReq,
    PauseGenerationReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    TokenizerWorker,
    _handle_output_by_index,
    get_tokenizer_worker_class,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class CustomTokenizerWorker(TokenizerWorker):
    pass


class NotAWorker:
    pass


class DefaultServerArgs:
    def get_tokenizer_worker_class(self):
        return TokenizerWorker


class CustomServerArgs:
    def get_tokenizer_worker_class(self):
        return CustomTokenizerWorker


class InvalidServerArgs:
    def get_tokenizer_worker_class(self):
        return NotAWorker


def _make_batch_str_output() -> BatchStrOutput:
    return BatchStrOutput(
        rids=["rid-0", "rid-1"],
        spec_verify_ct=[0, 0],
        spec_num_correct_drafts=[0, 0],
        spec_correct_drafts_histogram=[[], []],
        finished_reasons=[None, {"type": "length"}],
        output_strs=["first", "second"],
        output_ids=[[1], [2]],
        prompt_tokens=[10, 20],
        completion_tokens=[1, 2],
        reasoning_tokens=[0, 0],
        cached_tokens=[3, 4],
        cached_tokens_details=[
            {"device": 3, "host": 0},
            {"device": 1, "host": 3},
        ],
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=[0.0, 0.0],
        output_token_sampling_mask=[[], []],
        output_token_sampling_logprobs=[[], []],
        output_hidden_states=[None, None],
        routed_experts=[None, None],
        indexer_topk=[None, None],
        placeholder_tokens_idx=[None, None],
        placeholder_tokens_val=[None, None],
        retraction_counts=[0, 0],
        weight_versions=[
            [
                WeightVersionSpan(version="v1", start=0, end=3),
                WeightVersionSpan(version="v2", start=3, end=5),
            ],
            [WeightVersionSpan(version="v2", start=0, end=2)],
        ],
    )


class TestMultiTokenizerMixin(unittest.TestCase):
    def test_batch_str_output_preserves_cached_tokens_details(self):
        output = _make_batch_str_output()

        single_output = _handle_output_by_index(output, 1)

        self.assertEqual(single_output.rids, ["rid-1"])
        self.assertEqual(single_output.cached_tokens, [4])
        self.assertEqual(
            single_output.cached_tokens_details,
            [{"device": 1, "host": 3}],
        )

    def test_batch_str_output_keeps_weight_versions_nested_per_request(self):
        """Per-request segment lists stay one level nested after the split."""
        output = _make_batch_str_output()

        self.assertEqual(
            _handle_output_by_index(output, 0).weight_versions,
            [
                [
                    WeightVersionSpan(version="v1", start=0, end=3),
                    WeightVersionSpan(version="v2", start=3, end=5),
                ]
            ],
        )
        self.assertEqual(
            _handle_output_by_index(output, 1).weight_versions,
            [[WeightVersionSpan(version="v2", start=0, end=2)]],
        )

    def test_batch_str_output_without_weight_versions_stays_none(self):
        """An output from an older server without the field splits into None."""
        output = _make_batch_str_output()
        output.weight_versions = None

        self.assertIsNone(_handle_output_by_index(output, 0).weight_versions)

    def test_get_tokenizer_worker_class_uses_default(self):
        self.assertIs(get_tokenizer_worker_class(DefaultServerArgs()), TokenizerWorker)

    def test_get_tokenizer_worker_class_resolves_custom_class(self):
        self.assertIs(
            get_tokenizer_worker_class(CustomServerArgs()),
            CustomTokenizerWorker,
        )

    def test_get_tokenizer_worker_class_rejects_non_worker(self):
        with self.assertRaisesRegex(TypeError, "TokenizerWorker"):
            get_tokenizer_worker_class(InvalidServerArgs())


class TestPauseContinueWaiters(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.worker = TokenizerWorker.__new__(TokenizerWorker)
        self.worker.is_pause = False
        self.worker.is_pause_cond = asyncio.Condition()
        self.worker.model_update_lock = Mock()
        self.dispatched = []
        self.worker._dispatch_to_scheduler = self.dispatched.append
        self.worker._pause_continue_futures = {}

    async def _start_concurrent_operations(self):
        pause_task = asyncio.create_task(
            self.worker.pause_generation(PauseGenerationReqInput(mode="retract"))
        )
        continue_task = asyncio.create_task(
            self.worker.continue_generation(ContinueGenerationReqInput())
        )
        await asyncio.sleep(0)
        self.assertEqual(len(self.dispatched), 2)
        return pause_task, continue_task

    async def test_concurrent_reverse_broadcasts_resolve_matching_waiters(self):
        pause_task, continue_task = await self._start_concurrent_operations()
        pause_req, continue_req = self.dispatched

        await self.worker._apply_pause_continue_broadcast(
            PauseContinueBroadcastReq(
                is_pause=False, operation_id=continue_req.operation_id
            )
        )
        await asyncio.sleep(0)
        self.assertTrue(continue_task.done())
        self.assertFalse(pause_task.done())

        await self.worker._apply_pause_continue_broadcast(
            PauseContinueBroadcastReq(is_pause=True, operation_id=pause_req.operation_id)
        )
        await asyncio.wait_for(asyncio.gather(pause_task, continue_task), timeout=1)
        self.assertTrue(self.worker.is_pause)
        self.assertEqual(self.worker._pause_continue_futures, {})

    async def test_cancelled_operation_removes_its_waiter(self):
        task = asyncio.create_task(
            self.worker.continue_generation(ContinueGenerationReqInput())
        )
        await asyncio.sleep(0)
        operation_id = self.dispatched[0].operation_id

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertNotIn(operation_id, self.worker._pause_continue_futures)
        await self.worker._apply_pause_continue_broadcast(
            PauseContinueBroadcastReq(is_pause=True, operation_id=operation_id)
        )
        self.assertTrue(self.worker.is_pause)
        self.assertEqual(self.worker._pause_continue_futures, {})

    async def test_dispatch_failure_removes_its_waiter(self):
        def fail_dispatch(_obj):
            raise RuntimeError("dispatch failed")

        self.worker._dispatch_to_scheduler = fail_dispatch
        with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
            await self.worker.pause_generation(PauseGenerationReqInput(mode="retract"))

        self.assertEqual(self.worker._pause_continue_futures, {})

    async def test_unknown_late_broadcast_only_updates_pause_state(self):
        await self.worker._apply_pause_continue_broadcast(
            PauseContinueBroadcastReq(is_pause=True, operation_id="already-finished")
        )

        self.assertTrue(self.worker.is_pause)
        self.assertEqual(self.worker._pause_continue_futures, {})


if __name__ == "__main__":
    unittest.main()
