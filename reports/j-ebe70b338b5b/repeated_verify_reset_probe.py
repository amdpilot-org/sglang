"""Bounded MI300X probe for repeated Mamba verify/reset destinations."""

import itertools
import json
import time

import torch
from types import SimpleNamespace

from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import CaptureHiddenMode, ScheduleBatch
from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify


def _make_req(next_index):
    return SimpleNamespace(
        kv=SimpleNamespace(mamba_next_track_idx=next_index),
        finished=lambda: False,
        return_logprob=False,
        grammar=None,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
    )


def _make_batch(plans, mapping, request_indices):
    batch = ScheduleBatch(
        reqs=[_make_req(plan) for plan in plans],
        device="cuda",
    )
    batch.req_pool_indices = torch.tensor(request_indices, device="cuda")
    batch.req_pool_indices_cpu = torch.tensor(request_indices, dtype=torch.int64)
    batch.seq_lens = torch.arange(1, len(plans) + 1, device="cuda")
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.multimodal_inputs = None
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.sampling_info = SimpleNamespace(filter_batch=lambda *args, **kwargs: None)
    batch.spec_info = None
    batch.req_to_token_pool = SimpleNamespace(
        req_index_to_mamba_ping_pong_track_buffer_mapping=mapping
    )
    batch.mamba_lazy_spec_track_positions_cpu = list(plans)
    batch.mamba_track_mask = torch.ones(len(plans), dtype=torch.bool, device="cuda")
    batch.mamba_track_seqlens = torch.ones(
        len(plans), dtype=torch.int64, device="cuda"
    )
    return batch


def _verify(batch, plans, mapping, label):
    batch.req_to_token_pool.req_index_to_mamba_ping_pong_track_buffer_mapping = mapping
    batch.mamba_lazy_spec_track_positions_cpu = list(plans)
    batch.mamba_track_mask = torch.ones(
        len(batch.reqs), dtype=torch.bool, device="cuda"
    )
    batch.mamba_track_seqlens = torch.ones(
        len(batch.reqs), dtype=torch.int64, device="cuda"
    )
    print(
        f"{label}: req_pool_indices={batch.req_pool_indices.tolist()} "
        f"plans={list(plans)} mapping_rows={mapping.tolist()}",
        flush=True,
    )
    prepare_mamba_track_for_verify(batch)
    expected = [
        int(mapping[int(request_index), plan])
        for request_index, plan in zip(batch.req_pool_indices.tolist(), plans)
    ]
    actual = batch.mamba_track_indices.tolist()
    assert actual == expected, (label, actual, expected)
    assert batch.mamba_track_buffer_indices == list(plans)
    assert batch.mamba_track_mask is None
    assert batch.mamba_track_seqlens is None
    return actual


def run_case(case_index, initial_plans, first_keep):
    first_mapping = torch.tensor(
        [[100 + 2 * row, 101 + 2 * row] for row in range(6)],
        dtype=torch.int64,
        device="cuda",
    )
    batch = _make_batch(initial_plans, first_mapping, [0, 1, 2])
    first_destinations = _verify(
        batch, initial_plans, first_mapping, "verify-1"
    )

    batch.filter_batch(keep_indices=first_keep)
    assert batch.mamba_track_indices is None
    assert batch.mamba_track_buffer_indices is None
    assert batch.mamba_lazy_spec_track_positions_cpu is None

    second_plans = [1 - initial_plans[index] for index in first_keep]
    second_mapping = torch.tensor(
        [[200 + 2 * row, 201 + 2 * row] for row in range(6)],
        dtype=torch.int64,
        device="cuda",
    )
    second_destinations = _verify(
        batch, second_plans, second_mapping, "verify-2"
    )

    batch.filter_batch(keep_indices=[0])
    assert batch.mamba_track_indices is None
    assert batch.mamba_track_buffer_indices is None
    assert batch.mamba_lazy_spec_track_positions_cpu is None

    third_plans = [1 - second_plans[0]]
    third_mapping = torch.tensor(
        [[300 + 2 * row, 301 + 2 * row] for row in range(6)],
        dtype=torch.int64,
        device="cuda",
    )
    third_destinations = _verify(
        batch, third_plans, third_mapping, "verify-3"
    )

    return {
        "case": case_index,
        "initial_plans": list(initial_plans),
        "first_keep": list(first_keep),
        "second_plans": second_plans,
        "third_plan": third_plans,
        "destinations": [first_destinations, second_destinations, third_destinations],
    }


def main():
    torch.manual_seed(0)
    with get_context().override_server_args(
        mamba_radix_cache_strategy="extra_buffer_lazy"
    ):
        cases = []
        case_index = 0
        for initial_plans in itertools.product((0, 1), repeat=3):
            for first_keep in ([0, 1], [1, 2]):
                cases.append(run_case(case_index, initial_plans, first_keep))
                case_index += 1

        print(json.dumps({"status": "passed", "cases": cases}, indent=2))


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"ELAPSED_SECONDS={time.perf_counter() - started:.6f}")
