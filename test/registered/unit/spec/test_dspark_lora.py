"""CPU contracts for the DSpark draft-only LoRA path."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_lora import init_dspark_lora_manager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDSparkLoRAManager(CustomTestCase):
    @patch("sglang.srt.speculative.dspark_lora.LoRAManager")
    @patch("sglang.srt.speculative.dspark_lora.get_lora")
    @patch("sglang.srt.speculative.dspark_lora.get_spec")
    def test_adapter_is_draft_local_pinned_and_resident(
        self, get_spec, get_lora, manager_cls
    ):
        get_spec.return_value.speculative_dspark_lora_path = "/adapter/domain-a"
        get_lora.return_value.lora_backend = "torch_native"
        manager = manager_cls.return_value
        runner = SimpleNamespace(
            model=object(),
            model_config=SimpleNamespace(hf_config=object()),
            load_config=object(),
            dtype=torch.float16,
            server_args=object(),
            ps=SimpleNamespace(tp_size=1, tp_rank=0),
        )

        actual_manager, adapter_id = init_dspark_lora_manager(runner)

        self.assertIs(actual_manager, manager)
        kwargs = manager_cls.call_args.kwargs
        self.assertEqual(kwargs["max_loras_per_batch"], 2)
        self.assertEqual(kwargs["lora_backend"], "torch_native")
        ref = kwargs["lora_paths"][0]
        self.assertIsInstance(ref, LoRARef)
        self.assertEqual(ref.lora_path, "/adapter/domain-a")
        self.assertTrue(ref.pinned)
        self.assertEqual(adapter_id, ref.lora_id)
        manager.fetch_new_loras.assert_called_once_with({adapter_id})


class TestDSparkLoRARouting(CustomTestCase):
    def test_every_draft_request_routes_to_fixed_adapter(self):
        proposer = DraftBlockProposer.__new__(DraftBlockProposer)
        proposer.gamma = 1
        proposer.query_token_num = 1
        proposer._mask_token_id = 7
        proposer._draft_block_ids_buf = None
        proposer._draft_block_spec_info = object()
        proposer._dp_moe_sync = False
        proposer._num_token_non_padded = None
        proposer.draft_model = object()
        proposer.sample_from_anchor = True

        manager = Mock()
        raw_hidden = torch.zeros((2, 1))
        runner = SimpleNamespace(
            device=torch.device("cpu"),
            lora_manager=manager,
            dspark_lora_id="draft-adapter-id",
            decode_cuda_graph_runner=None,
            forward=Mock(
                return_value=SimpleNamespace(
                    logits_output=SimpleNamespace(hidden_states=raw_hidden),
                    can_run_graph=False,
                )
            ),
        )
        proposer.draft_model_runner = runner
        batch = SimpleNamespace(
            seq_lens=torch.tensor([3, 5]),
            seq_lens_cpu=torch.tensor([3, 5]),
            req_pool_indices=torch.tensor([0, 1]),
            can_run_decode_cuda_graph=False,
        )
        draft_input = SimpleNamespace(
            bonus_tokens=torch.tensor([1, 2]),
            nxt_kv_lens_cpu=None,
        )
        verify_window = SimpleNamespace(
            positions_2d=torch.tensor([[3], [5]]),
            verify_cache_loc_2d=torch.tensor([[10], [11]]),
        )

        with patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=False,
        ):
            proposer._run_forward(
                batch=batch,
                draft_input=draft_input,
                verify_window=verify_window,
                bs=2,
                device="cpu",
                embed_module=lambda ids: torch.zeros((*ids.shape, 1)),
            )

        forward_batch = runner.forward.call_args.args[0]
        self.assertEqual(forward_batch.lora_ids, ["draft-adapter-id"] * 2)
        manager.prepare_lora_batch.assert_called_once_with(forward_batch)


if __name__ == "__main__":
    unittest.main()
