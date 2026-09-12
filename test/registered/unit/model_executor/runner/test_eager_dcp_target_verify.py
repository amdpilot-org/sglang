from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.eager_runner import _get_dcp_extend_metadata


def _target_verify_batch(*, seq_lens, seq_lens_cpu, live_seq_lens_cpu):
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens_cpu,
        seq_lens_sum=999,
        extend_prefix_lens=None,
        extend_prefix_lens_cpu=None,
        extend_seq_lens=None,
        spec_info=SimpleNamespace(
            draft_token_num=7,
            live_seq_lens_cpu=live_seq_lens_cpu,
        ),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dcp_target_verify_derives_prefix_and_verify_geometry(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    prefix = torch.tensor([11, 23], dtype=torch.int32, device=device)
    batch = _target_verify_batch(
        seq_lens=prefix,
        # DSpark temporarily expands this mirror before ForwardBatch creation.
        seq_lens_cpu=torch.tensor([18, 30], dtype=torch.int32),
        live_seq_lens_cpu=torch.tensor([11, 23], dtype=torch.int32),
    )

    seq_lens, prefix_lens, prefix_lens_cpu, extend_lens, seq_lens_sum = (
        _get_dcp_extend_metadata(batch)
    )

    torch.testing.assert_close(prefix_lens, prefix)
    assert prefix_lens_cpu == [11, 23]
    torch.testing.assert_close(
        extend_lens, torch.tensor([7, 7], dtype=torch.int32, device=device)
    )
    torch.testing.assert_close(
        seq_lens, torch.tensor([18, 30], dtype=torch.int32, device=device)
    )
    assert seq_lens_sum == 48


def test_dcp_target_verify_uses_committed_prefix_without_live_lengths():
    prefix = torch.tensor([3], dtype=torch.int32)
    batch = _target_verify_batch(
        seq_lens=prefix,
        # run_non_compact can install nxt_kv_lens_cpu here after capturing a
        # missing live host mirror.  This is an expanded total, not a prefix.
        seq_lens_cpu=torch.tensor([10], dtype=torch.int32),
        live_seq_lens_cpu=None,
    )

    _, _, prefix_lens_cpu, extend_lens, seq_lens_sum = _get_dcp_extend_metadata(batch)

    assert prefix_lens_cpu == [3]
    assert extend_lens.tolist() == [7]
    assert seq_lens_sum == 10


def test_dcp_target_verify_supports_gpu_only_sequence_lengths():
    prefix = torch.tensor([4, 9], dtype=torch.int32)
    batch = _target_verify_batch(
        seq_lens=prefix,
        seq_lens_cpu=None,
        live_seq_lens_cpu=None,
    )

    _, _, prefix_lens_cpu, extend_lens, seq_lens_sum = _get_dcp_extend_metadata(batch)

    assert prefix_lens_cpu == [4, 9]
    assert extend_lens.tolist() == [7, 7]
    assert seq_lens_sum == 27


def test_non_target_verify_metadata_is_unchanged():
    prefix = torch.tensor([5], dtype=torch.int32)
    extend = torch.tensor([2], dtype=torch.int32)
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor([7], dtype=torch.int32),
        seq_lens_sum=7,
        extend_prefix_lens=prefix,
        extend_prefix_lens_cpu=[5],
        extend_seq_lens=extend,
        spec_info=None,
    )

    assert _get_dcp_extend_metadata(batch) == (
        batch.seq_lens,
        prefix,
        [5],
        extend,
        7,
    )
