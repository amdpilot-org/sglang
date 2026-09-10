"""Validate the ROCm DSA TileLang DCP decode return contract."""

import unittest
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    is_fp8_fnuz,
    tilelang_sparse_fwd,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm as forward_mla_rocm_module,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
    DeepseekMLARocmForwardMixin,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.srt.utils import is_hip
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_RUNNABLE = is_hip() and torch.cuda.is_available()


class _FakeForwardMode:
    def is_decode(self):
        return True

    def is_target_verify(self):
        return False


class _FakeForwardBatch:
    forward_mode = _FakeForwardMode()


class _FakeParallel:
    dcp_enabled = True
    attn_dcp_size = 1
    dcp_comm_backend = "a2a"
    dcp_group = object()


class _FakeAttention:
    current_attention_backend = "dsa"
    kv_lora_rank = 512
    num_local_heads = 4
    v_head_dim = 512
    use_deep_gemm_bmm = False
    next_skip_topk = None

    def _skip_rope_for_dsa_tilelang_fused(self):
        return False

    def _fuse_rope_for_trtllm_mla(self, forward_batch):
        return False

    def attn_mqa_for_dcp_decode(self, *args, **kwargs):
        return torch.zeros(1, 1, 4, 512), torch.zeros(1, 4)

    def o_proj(self, attn_bmm_output):
        return attn_bmm_output, None


@unittest.skipUnless(_RUNNABLE, "requires an available HIP GPU")
class TestDSATilelangDCPLse(CustomTestCase):
    def _reference(self, q, kv, dim, scale):
        scores = torch.einsum("bhd,nkd->bhn", q.float(), kv.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("bhn,nkd->bhd", weights, kv[:, :, :dim].float())
        lse = torch.logsumexp(scores, dim=-1) / torch.log(
            torch.tensor(2.0, device=q.device)
        )
        return output, lse

    def test_bf16_output_and_base2_lse(self):
        torch.manual_seed(38709)
        device = "cuda"
        batch, heads, dim, tail, topk = 1, 4, 512, 64, 64
        scale = dim**-0.5
        q = torch.randn(batch, heads, dim + tail, device=device, dtype=torch.bfloat16)
        kv = torch.randn(topk, 1, dim + tail, device=device, dtype=torch.bfloat16)
        indices = torch.arange(topk, device=device, dtype=torch.int32).reshape(
            1, 1, topk
        )

        single = tilelang_sparse_fwd(q, kv, indices, scale, dim, return_lse=False)
        output, lse = tilelang_sparse_fwd(q, kv, indices, scale, dim, return_lse=True)
        ref_output, ref_lse = self._reference(q, kv, dim, scale)

        self.assertIsInstance(single, torch.Tensor)
        self.assertIsInstance(output, torch.Tensor)
        self.assertIsInstance(lse, torch.Tensor)
        self.assertTrue(torch.equal(single, output))
        self.assertEqual(tuple(lse.shape), (batch, heads))
        self.assertLessEqual((output.float() - ref_output).abs().max().item(), 3e-3)
        self.assertLessEqual((lse - ref_lse).abs().max().item(), 1e-5)

    def test_fp8_kv_output_and_base2_lse(self):
        torch.manual_seed(38709)
        device = "cuda"
        batch, heads, dim, tail, topk = 1, 4, 512, 64, 64
        scale = dim**-0.5
        fp8_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz else torch.float8_e4m3fn
        q = torch.randn(batch, heads, dim + tail, device=device).to(fp8_dtype)
        kv = torch.randn(topk, 1, dim + tail, device=device).to(fp8_dtype)
        indices = torch.arange(topk, device=device, dtype=torch.int32).reshape(
            1, 1, topk
        )

        output, lse = tilelang_sparse_fwd(q, kv, indices, scale, dim, return_lse=True)
        ref_output, ref_lse = self._reference(q, kv, dim, scale)

        self.assertEqual(tuple(output.shape), (batch, batch, heads, dim))
        self.assertEqual(tuple(lse.shape), (batch, heads))
        self.assertLessEqual((output.float() - ref_output).abs().max().item(), 2e-2)
        self.assertLessEqual((lse - ref_lse).abs().max().item(), 1e-5)

    def test_rocm_dcp_decode_caller_unpacks_tuple(self):
        attention = _FakeAttention()
        forward_batch = _FakeForwardBatch()
        merged = {}

        def fake_reduce(output, lse, *args, **kwargs):
            merged["output"] = output
            merged["lse"] = lse
            return output

        patches = {
            "is_dcp_mla_decode_phase": mock.Mock(return_value=True),
            "get_parallel": mock.Mock(return_value=_FakeParallel()),
            "get_in_autotune_dummy_run": mock.Mock(return_value=False),
            "is_mla_dcp_lse_base_on_e": mock.Mock(return_value=False),
            "dcp_a2a_lse_reduce": fake_reduce,
            "rocm_absorb_v_bmm": mock.Mock(
                side_effect=lambda self, output: output.flatten(1)
            ),
            "is_kv_b_lora_active": mock.Mock(return_value=False),
        }
        with mock.patch.multiple(
            forward_mla_rocm_module,
            **{name: value for name, value in patches.items()},
        ):
            output = DeepseekMLARocmForwardMixin.forward_absorb_rocm_core(
                attention,
                q_pe=None,
                k_pe=None,
                q_nope_out=None,
                k_nope=None,
                forward_batch=forward_batch,
                zero_allocator=None,
                positions=None,
                topk_indices=None,
                llama_4_scaling=None,
            )

        self.assertEqual(output.shape, (1, 4 * 512))
        self.assertEqual(merged["output"].shape, (1, 4, 512))
        self.assertEqual(merged["lse"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main(verbosity=3)
