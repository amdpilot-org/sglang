from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    RaggedVerifyMode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_TIER = 192
_WIDTH = 6


def _layout(verify_lens):
    return RaggedVerifyLayout.from_verify_lens(
        verify_lens_cpu=verify_lens,
        device=torch.device("cpu"),
        grid=[1, 32, _TIER],
    )


def _runner_with_captured_layout():
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.captured_req_width = _WIDTH
    captured = _layout([1] * _TIER)
    runner._captured_ragged_layouts = {_TIER: captured}
    return runner, captured


@pytest.mark.parametrize(
    "live_lens, expected_lens",
    [
        ([6] * 32, [6] * 32 + [0] * 160),
        # 160 live tokens select the same 192-token tier. The remaining graph
        # tokens must be assigned to synthetic request slots, not discarded.
        ([5] * 32, [5] * 32 + [1] * 32 + [0] * 128),
    ],
)
def test_runner_stages_live_geometry_into_pointer_stable_capture_buffers(
    live_lens, expected_lens
):
    runner, captured = _runner_with_captured_layout()
    verify_lens_ptr = captured.verify_lens.data_ptr()
    qo_indptr_ptr = captured.qo_indptr_device.data_ptr()

    runner._stage_ragged_verify_layout(_layout(live_lens), _TIER)

    assert captured.verify_lens.tolist() == expected_lens
    assert captured.qo_indptr_device.tolist() == torch.nn.functional.pad(
        torch.tensor(expected_lens, dtype=torch.int32), (1, 0)
    ).cumsum(0).tolist()
    assert captured.verify_lens.data_ptr() == verify_lens_ptr
    assert captured.qo_indptr_device.data_ptr() == qo_indptr_ptr


def test_dsv4_replay_resolves_the_runner_staged_layout():
    runner, captured = _runner_with_captured_layout()
    runner._stage_ragged_verify_layout(_layout([6] * 32), _TIER)
    forward_batch = SimpleNamespace(
        spec_info=SimpleNamespace(ragged_verify_layout=captured)
    )
    backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
    backend.online_c128_mtp = SimpleNamespace(enabled=lambda: False)

    with (
        patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.read_ragged_verify_mode",
            return_value=RaggedVerifyMode.COMPACT,
        ),
        patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.get_parallel",
            return_value=SimpleNamespace(attn_cp_size=1),
        ),
    ):
        replay_layout = backend._resolve_verify_layout(forward_batch, bs=_TIER)

    assert replay_layout.verify_lens.tolist() == [6] * 32 + [0] * 160
    assert replay_layout.qo_indptr_device[-1].item() == _TIER


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
