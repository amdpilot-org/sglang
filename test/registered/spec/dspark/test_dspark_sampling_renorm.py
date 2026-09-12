import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.speculative import dflash_utils
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.srt.utils import is_hip


register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="base-b", runner_config="1-gpu-small-amd")


def _reference_top_k_top_p(logits, temperatures, top_ks, top_ps, draft_token_num):
    temperatures = torch.repeat_interleave(temperatures, draft_token_num, dim=0)
    top_ks = torch.repeat_interleave(top_ks, draft_token_num, dim=0)
    top_ps = torch.repeat_interleave(top_ps, draft_token_num, dim=0)
    probs = torch.softmax(logits / temperatures, dim=-1)

    result = torch.zeros_like(probs)
    for row in range(probs.shape[0]):
        k = max(1, min(int(top_ks[row]), probs.shape[1]))
        values, indices = torch.sort(probs[row], descending=True)
        values[k:] = 0
        values /= values.sum()
        cumulative = values.cumsum(dim=0)
        values[(cumulative - values) > top_ps[row]] = 0
        values /= values.sum()
        result[row].scatter_(0, indices, values)
    return result


class TestDSparkSamplingRenorm(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and is_hip(), "requires ROCm")
    def test_rocm_uses_working_triton_renormalizers(self):
        probs = torch.tensor(
            [[0.05, 0.15, 0.5, 0.2, 0.1], [0.4, 0.1, 0.2, 0.05, 0.25]],
            device="cuda",
        )
        top_ks = torch.tensor([1, 5], device="cuda", dtype=torch.int32)
        top_ps = torch.tensor([1.0, 0.6], device="cuda")

        self.assertIsNotNone(dflash_utils.top_k_renorm_prob)
        self.assertIsNotNone(dflash_utils.top_p_renorm_prob)
        top_k_actual = dflash_utils._dflash_top_k_renorm_prob(probs, top_ks)
        top_p_actual = dflash_utils._dflash_top_p_renorm_prob(probs, top_ps)

        top_k_expected = torch.tensor(
            [[0.0, 0.0, 1.0, 0.0, 0.0], [0.4, 0.1, 0.2, 0.05, 0.25]],
            device="cuda",
        )
        # Row 1 keeps 0.4 and 0.25 because the mass before 0.25 is below 0.6.
        top_p_expected = torch.tensor(
            [[0.05, 0.15, 0.5, 0.2, 0.1], [0.4 / 0.65, 0, 0, 0, 0.25 / 0.65]],
            device="cuda",
        )
        torch.testing.assert_close(top_k_actual, top_k_expected)
        torch.testing.assert_close(top_p_actual, top_p_expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_missing_optional_kernels_fall_back_for_top_k_and_top_p(self):
        """The DSPARK verify path must not call None on non-CUDA backends."""
        device = torch.device("cuda")
        draft_token_num = 2
        batch_size = 3
        logits = torch.tensor(
            [
                [3.0, 2.0, 1.0, 0.0, -1.0],
                [0.0, 1.0, 3.0, 2.0, -1.0],
                [2.0, -2.0, 0.0, 1.0, 3.0],
                [-1.0, 0.0, 1.0, 2.0, 3.0],
                [1.0, 4.0, 0.0, 3.0, 2.0],
                [4.0, 1.0, 3.0, 0.0, 2.0],
            ],
            device=device,
        )
        sampling_info = types.SimpleNamespace(
            need_top_k_sampling=True,
            need_top_p_sampling=True,
            temperatures=torch.tensor([[1.0], [0.7], [1.3]], device=device),
            # Covers k=1, an interior k, and k above the vocabulary size.
            top_ks=torch.tensor([1, 3, 99], device=device, dtype=torch.int32),
            # Covers active filtering and the no-op boundary.
            top_ps=torch.tensor([0.55, 0.8, 1.0], device=device),
        )

        expected = _reference_top_k_top_p(
            logits,
            sampling_info.temperatures,
            sampling_info.top_ks,
            sampling_info.top_ps,
            draft_token_num,
        ).view(batch_size, draft_token_num, -1)

        with (
            patch.object(dflash_utils, "top_k_renorm_prob", None),
            patch.object(dflash_utils, "top_p_renorm_prob", None),
        ):
            actual = dflash_utils.build_dflash_verify_target_probs(
                next_token_logits=logits,
                sampling_info=sampling_info,
                draft_token_num=draft_token_num,
                bs=batch_size,
                use_sparse_topk=False,
            )

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(
            actual.sum(dim=-1), torch.ones_like(actual.sum(dim=-1))
        )


if __name__ == "__main__":
    unittest.main()
