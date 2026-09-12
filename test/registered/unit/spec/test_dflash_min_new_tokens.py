"""DFLASH/DSPARK regression coverage for min-new-token penalty lifecycle."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.sampling.penaltylib.min_new_tokens import (
    BatchedMinNewTokensPenalizer,
)
from sglang.srt.sampling.penaltylib.orchestrator import (
    BatchedPenalizerOrchestrator,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SamplingInfo(SimpleNamespace):
    def __len__(self):
        return len(self.temperatures)


class _Batch(SimpleNamespace):
    def batch_size(self):
        return len(self.reqs)

    def maybe_evict_swa(self):
        pass

    def cumulate_penalty_output_tokens(self):
        ScheduleBatch.cumulate_penalty_output_tokens(self)


def _make_batch(min_new_tokens=2):
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            min_new_tokens=min_new_tokens,
            stop_token_ids={3},
            top_k=1,
        ),
        eos_token_ids={2},
        tokenizer=SimpleNamespace(
            additional_stop_token_ids=set(),
            eos_token_id=2,
        ),
        output_ids=[7],
        origin_input_ids=[1],
        kv=SimpleNamespace(kv_committed_len=1),
        decode_batch_idx=0,
    )
    batch = _Batch(
        reqs=[req],
        device="cpu",
        tree_cache=None,
        req_to_token_pool=None,
        req_pool_indices=torch.tensor([0]),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
    )
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size=8,
        batch=batch,
        penalizers={BatchedMinNewTokensPenalizer},
    )
    batch.sampling_info = _SamplingInfo(
        temperatures=torch.ones((1, 1)),
        vocab_size=8,
        penalizer_orchestrator=orchestrator,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
    )
    return batch


def _refresh_penalties(batch):
    SamplingBatchInfo.update_penalties(batch.sampling_info)
    return batch.sampling_info.acc_additive_penalties[0, 2].item()


def test_dflash_decode_advances_refreshes_and_removes_min_token_penalty():
    batch = _make_batch(min_new_tokens=2)
    draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))

    assert _refresh_penalties(batch) == float("-inf")

    spec = SimpleNamespace(speculative_num_draft_tokens=1)
    with (
        patch("sglang.srt.speculative.dflash_info_v2.get_spec", return_value=spec),
        patch(
            "sglang.srt.speculative.dflash_info_v2.page_aligned_decode_alloc_lens",
            return_value=([1], [2], 1),
        ),
        patch("sglang.srt.speculative.dflash_info_v2.alloc_for_spec_decode"),
    ):
        draft_input.prepare_for_decode(batch)
        assert _refresh_penalties(batch) == float("-inf")

        batch.reqs[0].output_ids.append(8)
        draft_input.prepare_for_decode(batch)
        assert _refresh_penalties(batch) == 0.0

    batch.reqs[0].sampling_params.min_new_tokens = 0
    batch.sampling_info.penalizer_orchestrator.filter(torch.tensor([0]))
    SamplingBatchInfo.update_penalties(batch.sampling_info)
    assert batch.sampling_info.acc_additive_penalties is None
