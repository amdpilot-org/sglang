import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase


register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


def load_common_utils():
    repo_root = Path(__file__).resolve().parents[5]
    module_path = (
        repo_root
        / "benchmark/kernels/fused_moe_triton/common_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "fused_moe_tuning_common_utils", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFusedMoeTunerValidation(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/HIP is not available")
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        self.common_utils = load_common_utils()
        self.x = torch.empty(8, 64, dtype=torch.bfloat16, device="cuda").normal_(
            0, 0.01
        )
        self.w1 = torch.empty(
            8, 128, 64, dtype=torch.bfloat16, device="cuda"
        ).normal_(0, 0.01)
        self.w2 = torch.empty(
            8, 64, 64, dtype=torch.bfloat16, device="cuda"
        ).normal_(0, 0.01)
        gating = torch.randn(8, 8, dtype=torch.float32, device="cuda")
        self.topk_output = select_experts(
            self.x, gating, TopKConfig(top_k=2, renormalize=True)
        )
        self.config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
            "waves_per_eu": 0,
        }

    def test_invalid_output_is_rejected(self):
        self.common_utils.validate_unquantized_moe_output(
            self.config, self.x, self.w1, self.w2, self.topk_output
        )
        expected = self.common_utils.reference_unquantized_moe(
            self.x,
            self.w1,
            self.w2,
            self.topk_output.topk_weights,
            self.topk_output.topk_ids,
        )
        invalid_output = expected + 1
        with mock.patch.object(
            self.common_utils, "fused_moe", return_value=invalid_output
        ):
            with self.assertRaises(AssertionError):
                self.common_utils.validate_unquantized_moe_output(
                    self.config, self.x, self.w1, self.w2, self.topk_output
                )


if __name__ == "__main__":
    unittest.main()
