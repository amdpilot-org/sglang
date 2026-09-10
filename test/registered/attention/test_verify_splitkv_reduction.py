"""Independent reduction/merge correctness tests for ``VerifySplitKV``.

These tests intentionally avoid ``extend_attention_fwd``. They use a direct
fp32 Torch reference to check the split-prefix reduction, causal draft mask,
and final LSE merge for every supported split count on ragged final partitions.
"""

import statistics
import unittest

import torch

from sglang.kernels.ops.attention.verify_splitkv import VerifySplitKV
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")


ATOL = 2e-2
RTOL = 1e-2


def _build_verify_inputs(
    prefix_lens,
    l_ext,
    h_q,
    h_kv,
    head_dim,
    v_head_dim,
    dtype,
    device,
):
    """Build a deterministic verify-shaped problem with ragged prefix lengths."""
    batch_size = len(prefix_lens)
    prefix_lens_t = torch.tensor(prefix_lens, dtype=torch.int32, device=device)
    total_prefix = int(prefix_lens_t.sum())

    k_buffer = torch.randn(
        total_prefix, h_kv, head_dim, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        total_prefix, h_kv, v_head_dim, dtype=dtype, device=device
    )
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(prefix_lens_t, dim=0)
    kv_indices = torch.arange(total_prefix, dtype=torch.int64, device=device)

    n_ext = batch_size * l_ext
    q_extend = torch.randn(n_ext, h_q, head_dim, dtype=dtype, device=device)
    k_extend = torch.randn(n_ext, h_kv, head_dim, dtype=dtype, device=device)
    v_extend = torch.randn(
        n_ext, h_kv, v_head_dim, dtype=dtype, device=device
    )
    qo_indptr = torch.arange(
        0, n_ext + 1, l_ext, dtype=torch.int32, device=device
    )

    return (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        l_ext,
    )


def _independent_torch_reference(
    q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    k_scale,
    v_scale,
    is_causal=True,
):
    """Compute verify attention in fp32 with an explicit prefix/draft LSE merge."""
    batch_size = qo_indptr.shape[0] - 1
    head_dim = q_extend.shape[2]
    v_head_dim = v_extend.shape[2]
    h_q = q_extend.shape[1]
    h_kv = k_extend.shape[1]
    group_size = h_q // h_kv
    output = torch.empty(
        q_extend.shape[0],
        h_q,
        v_head_dim,
        dtype=q_extend.dtype,
        device=q_extend.device,
    )

    for batch in range(batch_size):
        q_start = int(qo_indptr[batch])
        q_end = int(qo_indptr[batch + 1])
        l_ext = q_end - q_start

        kv_start = int(kv_indptr[batch])
        kv_end = int(kv_indptr[batch + 1])
        prefix_indices = kv_indices[kv_start:kv_end]

        q_batch = q_extend[q_start:q_end].float()
        k_prefix = k_buffer[prefix_indices].float()
        v_prefix = v_buffer[prefix_indices].float()
        k_draft = k_extend[q_start:q_end].float()
        v_draft = v_extend[q_start:q_end].float()

        if is_causal:
            draft_mask = torch.tril(
                torch.ones(
                    l_ext,
                    l_ext,
                    dtype=torch.bool,
                    device=q_extend.device,
                )
            )
        else:
            draft_mask = torch.ones(
                l_ext,
                l_ext,
                dtype=torch.bool,
                device=q_extend.device,
            )

        for head in range(h_q):
            kv_head = head // group_size

            prefix_scores = (
                q_batch[:, head, :]
                @ k_prefix[:, kv_head, :].transpose(0, 1)
            ) * sm_scale * k_scale
            prefix_probs = torch.softmax(prefix_scores, dim=-1)
            prefix_output = (
                prefix_probs @ v_prefix[:, kv_head, :]
            ) * v_scale
            prefix_lse = torch.logsumexp(prefix_scores, dim=-1)

            draft_scores = (
                q_batch[:, head, :]
                @ k_draft[:, kv_head, :].transpose(0, 1)
            ) * sm_scale
            draft_scores = draft_scores.masked_fill(
                ~draft_mask, float("-inf")
            )
            draft_probs = torch.softmax(draft_scores, dim=-1)
            draft_output = draft_probs @ v_draft[:, kv_head, :]
            draft_lse = torch.logsumexp(draft_scores, dim=-1)

            max_lse = torch.maximum(prefix_lse, draft_lse)
            prefix_weight = torch.exp(prefix_lse - max_lse)
            draft_weight = torch.exp(draft_lse - max_lse)
            merged_output = (
                prefix_output * prefix_weight[:, None]
                + draft_output * draft_weight[:, None]
            ) / (prefix_weight + draft_weight)[:, None]

            output[q_start:q_end, head] = merged_output.to(output.dtype)

    return output


def _warm_kernel_timing(run_kernel, iterations=5):
    """Return median warm kernel latency in milliseconds using CUDA events."""
    run_kernel()
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_kernel()
        end.record()
        torch.cuda.synchronize()
        timings_ms.append(start.elapsed_time(end))

    return statistics.median(timings_ms)


@unittest.skipIf(not torch.cuda.is_available(), "GPU required")
class TestVerifySplitKVReduction(CustomTestCase):
    def test_supported_split_counts_ragged_partitions_and_lse_merge(self):
        prefix_lens = [127, 4127, 8223]
        l_ext = 4
        h_q = 8
        h_kv = 2
        head_dim = 128
        v_head_dim = 128
        dtype = torch.bfloat16
        device = "cuda"
        sm_scale = 1.0 / (head_dim**0.5)
        k_scale = 0.5
        v_scale = 0.25

        torch.manual_seed(0)
        (
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            _,
        ) = _build_verify_inputs(
            prefix_lens,
            l_ext,
            h_q,
            h_kv,
            head_dim,
            v_head_dim,
            dtype,
            device,
        )

        reference = _independent_torch_reference(
            q_extend,
            k_extend,
            v_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            k_scale,
            v_scale,
            is_causal=True,
        )

        for n_splits in (4, 8, 16):
            with self.subTest(n_splits=n_splits):
                split_kv = VerifySplitKV(
                    max_bs=len(prefix_lens),
                    h_q=h_q,
                    h_kv=h_kv,
                    head_dim=head_dim,
                    v_head_dim=v_head_dim,
                    l_ext=l_ext,
                    device=device,
                    n_splits=n_splits,
                )
                split_output = torch.empty_like(reference)

                def run_kernel():
                    split_kv(
                        q_extend,
                        k_extend,
                        v_extend,
                        k_buffer,
                        v_buffer,
                        qo_indptr,
                        kv_indptr,
                        kv_indices,
                        sm_scale,
                        o_out=split_output,
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )

                run_kernel()
                torch.cuda.synchronize()

                max_abs_diff = (
                    (split_output.float() - reference.float())
                    .abs()
                    .max()
                    .item()
                )
                torch.testing.assert_close(
                    split_output,
                    reference,
                    atol=ATOL,
                    rtol=RTOL,
                )

                warm_ms = _warm_kernel_timing(run_kernel)
                print(
                    f"verify_splitkv reduction: n_splits={n_splits}, "
                    f"prefix_lens={prefix_lens}, max_abs_diff={max_abs_diff:.6g}, "
                    f"warm_median_ms={warm_ms:.6g}"
                )


if __name__ == "__main__":
    unittest.main()
