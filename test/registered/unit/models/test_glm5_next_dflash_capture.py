"""Regression for DFLASH aux-hidden capture on mHC models.

GLM-5.3-Flash runs with mhc=True. MHCLayerCommunicator folds the residual
into the widened hidden state and returns residual=None, so CUDA-graph
capture used to crash on `hidden_states + residual`. DFLASH also has to
contract that widened state back to the draft hidden size; skipping the
contract is a silent shape/quality bug the crash-guard alone would miss.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextDflashCapture(CustomTestCase):
    def _make_empty_pipeline_stage(self, *, mhc: bool) -> Glm5NextModel:
        model = Glm5NextModel.__new__(Glm5NextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(mhc=mhc)
        model.pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=False)
        model.start_layer = model.end_layer = 0
        model.first_k_dense_replace = 0
        model.dflash_capture = False
        model.layers_to_capture = []
        model.enable_a2a_moe = False
        return model

    def _forward_empty_pipeline_stage(self, model, proxy_tensors):
        return model.forward(
            input_ids=torch.empty(0, dtype=torch.long),
            positions=torch.empty(0, dtype=torch.long),
            forward_batch=SimpleNamespace(can_run_tbo=False),
            pp_proxy_tensors=PPProxyTensors(proxy_tensors),
        )

    def test_mhc_pipeline_stage_uses_folded_hidden_state_without_residual(self):
        model = self._make_empty_pipeline_stage(mhc=True)
        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        output = self._forward_empty_pipeline_stage(
            model, {"hidden_states": hidden_states}
        )

        self.assertEqual(set(output.tensors), {"hidden_states"})
        torch.testing.assert_close(output["hidden_states"], hidden_states)

    def test_non_mhc_pipeline_stage_preserves_separate_residual(self):
        model = self._make_empty_pipeline_stage(mhc=False)
        hidden_states = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        residual = torch.full_like(hidden_states, 2)

        output = self._forward_empty_pipeline_stage(
            model, {"hidden_states": hidden_states, "residual": residual}
        )

        self.assertEqual(set(output.tensors), {"hidden_states", "residual"})
        torch.testing.assert_close(output["hidden_states"], hidden_states)
        torch.testing.assert_close(output["residual"], residual)

    def test_dflash_contracts_mhc_hidden_state_without_residual(self):
        model = Glm5NextModel.__new__(Glm5NextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(mhc=True, hc_mult=4)
        model.dflash_capture = True

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        actual = model._prepare_aux_hidden_state(hidden_states, None)
        expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
