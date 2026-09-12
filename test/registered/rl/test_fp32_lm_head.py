import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


class TestFP32LogitsGatherFallback(unittest.TestCase):
    def test_fp32_uses_generic_all_gather_without_casting(self):
        gatherer = MultimemAllGatherer.__new__(MultimemAllGatherer)
        gatherer._state = None
        x = torch.randn(2, 8, dtype=torch.float32)

        with patch(
            "sglang.srt.distributed.tensor_model_parallel_all_gather",
            side_effect=lambda value, dim: torch.cat((value, value), dim=dim),
        ) as generic_gather:
            output = gatherer(x)

        generic_gather.assert_called_once()
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, torch.cat((x, x), dim=-1))


class LMHeadStub(nn.Module):
    def __init__(self, vocab, hidden, dtype, device=get_device()):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab, hidden, dtype=dtype, device=device)
        )


class DummyMeta:
    gathered_buffer = None
    next_token_logits_buffer = None

    def compute_dp_attention_metadata(self): ...


class TestLMHeadFP32(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() and not (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        ):
            raise unittest.SkipTest("needs CUDA GPU or XPU")

    def _make_logprocessor(self, vocab_size, enable_fp32, config_enable_fp32=False):
        # LogitsProcessor reads get_exec().features.enable_fp32_lm_head
        # from the published config.
        override = get_context().override_server_args(enable_fp32_lm_head=enable_fp32)
        override.install()
        self.addCleanup(override.restore)
        cfg = SimpleNamespace(
            vocab_size=vocab_size,
            final_logit_softcapping=None,
            enable_lm_head_fp32=config_enable_fp32,
        )
        return LogitsProcessor(cfg, skip_all_gather=True, logit_scale=None)

    def _run_case(
        self,
        hidden_state_dtype,
        enable_fp32,
        weights_dtype,
        expected_a_dtype,
        expected_b_dtype,
        expected_operation,
        config_enable_fp32=False,
    ):
        device = get_device()
        BATCH_SIZE, HIDDEN_SIZE, VOCAB_SIZE = 2, 64, 128
        hidden_state = torch.randn(
            BATCH_SIZE, HIDDEN_SIZE, dtype=hidden_state_dtype, device=device
        )
        head = LMHeadStub(VOCAB_SIZE, HIDDEN_SIZE, dtype=weights_dtype, device=device)
        meta = DummyMeta()
        logprocessor = self._make_logprocessor(
            VOCAB_SIZE, enable_fp32, config_enable_fp32
        )

        original_matmul = torch.matmul
        original_mm = torch.mm
        original_linear = F.linear

        state = {
            "called": False,  # Whether a matmul/linear call has been intercepted yet
            "operation": None,  # Which operation was captured ("matmul" or "linear")
            "a": None,  # The dtype of the first input tensor to the operation
            "b": None,  # The dtype of the second input tensor to the operation
            "out_dtype": None,
        }

        def probe_matmul(a, b, *args, **kw):
            if not state["called"]:
                state.update(
                    called=True,
                    operation="matmul",
                    a=a.dtype,
                    b=b.dtype,
                    out_dtype=kw.get("out_dtype"),
                )
            return original_matmul(a, b, *args, **kw)

        def probe_mm(a, b, *args, **kw):
            if not state["called"]:
                state.update(
                    called=True,
                    operation="mm",
                    a=a.dtype,
                    b=b.dtype,
                    out_dtype=kw.get("out_dtype"),
                )
            return original_mm(a, b, *args, **kw)

        def probe_linear(x, w, bias=None):
            if not state["called"]:
                state.update(called=True, ooperationp="linear", a=x.dtype, b=w.dtype)
            return original_linear(x, w, bias)

        with (
            patch("torch.matmul", new=probe_matmul),
            patch("torch.mm", new=probe_mm),
            patch("torch.nn.functional.linear", new=probe_linear),
        ):
            logits = logprocessor._get_logits(hidden_state, head, meta)
        self.assertEqual(hidden_state.dtype, hidden_state_dtype)
        self.assertTrue(state["called"], "no call lm head matlmul/linear")
        self.assertEqual(state["operation"], expected_operation)
        self.assertEqual(state["a"], expected_a_dtype)
        self.assertEqual(state["b"], expected_b_dtype)
        self.assertEqual(
            state["out_dtype"],
            torch.float32 if expected_operation == "mm" else None,
        )

    def test_flag_true_fp16_activations(self):
        expected_operation = "mm" if torch.cuda.is_available() else "matmul"
        expected_dtype = (
            torch.float32 if expected_operation == "matmul" else torch.float16
        )
        self._run_case(
            torch.float16,
            True,
            torch.float16,
            expected_dtype,
            expected_dtype,
            expected_operation,
        )

    def test_flag_true_bf16_activations(self):
        expected_operation = "mm" if torch.cuda.is_available() else "matmul"
        expected_dtype = (
            torch.float32 if expected_operation == "matmul" else torch.bfloat16
        )
        self._run_case(
            torch.bfloat16,
            True,
            torch.bfloat16,
            expected_dtype,
            expected_dtype,
            expected_operation,
        )

    def test_flag_true_fp32_falls_back_to_explicit_fp32_matmul(self):
        self._run_case(
            torch.float32,
            True,
            torch.float32,
            torch.float32,
            torch.float32,
            "matmul",
        )

    def test_flag_false_fp16_path(self):
        self._run_case(
            torch.float16, False, torch.float16, torch.float16, torch.float16, "matmul"
        )

    def test_flag_false_bf16_path(self):
        self._run_case(
            torch.bfloat16,
            False,
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            "matmul",
        )

    def test_model_config_enables_fp32_without_server_flag(self):
        expected_operation = "mm" if torch.cuda.is_available() else "matmul"
        expected_dtype = (
            torch.float32 if expected_operation == "matmul" else torch.bfloat16
        )
        self._run_case(
            torch.bfloat16,
            False,
            torch.bfloat16,
            expected_dtype,
            expected_dtype,
            expected_operation,
            config_enable_fp32=True,
        )

    def test_fp32_output_survives_tensor_parallel_gather(self):
        """The opt-in must not revert to BF16 merely because TP gathers logits."""
        device = get_device()
        hidden_size, local_vocab = 64, 32
        hidden_state = torch.randn(3, hidden_size, dtype=torch.bfloat16, device=device)
        head = LMHeadStub(local_vocab, hidden_size, torch.bfloat16, device=device)
        logprocessor = self._make_logprocessor(local_vocab * 2, enable_fp32=True)
        logprocessor.do_tensor_parallel_all_gather = True

        gathered_dtype = None

        def fake_all_gather(local_logits):
            nonlocal gathered_dtype
            gathered_dtype = local_logits.dtype
            return torch.cat((local_logits, local_logits), dim=-1)

        logprocessor._logits_gatherer = fake_all_gather
        logits = logprocessor._get_logits(hidden_state, head, DummyMeta())

        self.assertEqual(gathered_dtype, torch.float32)
        self.assertEqual(logits.dtype, torch.float32)
        reference = torch.matmul(hidden_state.float(), head.weight.float().T)
        torch.testing.assert_close(logits[:, :local_vocab], reference)
        torch.testing.assert_close(logits[:, local_vocab:], reference)

    def test_fp32_gemm_preserves_ranking_lost_by_bf16_output(self):
        """Compare both output choices to an independent explicit-FP32 reference."""
        if not torch.cuda.is_available():
            self.skipTest("torch.mm(out_dtype=...) requires CUDA or ROCm")

        device = get_device()
        hidden = torch.ones((1, 64), dtype=torch.bfloat16, device=device)
        weight = torch.zeros((2, 64), dtype=torch.bfloat16, device=device)
        weight[0].fill_(1)
        weight[1].fill_(1)
        # The exact BF16-input dot products differ, but both round to the same
        # BF16 logit. argmax then chooses row 0 instead of the true row 1.
        weight[1, 0] = torch.tensor(1.0078125, dtype=torch.bfloat16, device=device)

        reference = torch.matmul(hidden.float(), weight.float().T)
        direct_fp32 = torch.mm(hidden, weight.T, out_dtype=torch.float32)
        post_bf16_cast = torch.matmul(hidden, weight.T).float()

        self.assertEqual(reference.argmax(-1).item(), 1)
        self.assertEqual(direct_fp32.argmax(-1).item(), 1)
        self.assertEqual(post_bf16_cast.argmax(-1).item(), 0)
        torch.testing.assert_close(direct_fp32, reference, atol=0, rtol=0)
        self.assertGreater(
            (post_bf16_cast - reference).abs().max().item(),
            (direct_fp32 - reference).abs().max().item(),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
