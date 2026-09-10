"""GLM-5.3 Flash NextN embedding shard-index contract.

This test validates local shard indexing only.  It deliberately does not
initialize distributed process groups, so it cannot establish TP8 collective
behavior.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers import vocab_parallel_embedding as vpe
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)
from sglang.srt.runtime_context import get_flags, get_forward, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=13, stage="base-b-kernel-unit", runner_config="1-gpu-small")

VOCAB_SIZE = 8
HIDDEN_SIZE = 3
MODEL_TP_SIZE = 2
SHARD_SIZE = VOCAB_SIZE // MODEL_TP_SIZE


def _global_weight() -> torch.Tensor:
    return torch.arange(
        VOCAB_SIZE * HIDDEN_SIZE,
        dtype=torch.float32,
        device="cuda",
    ).reshape(VOCAB_SIZE, HIDDEN_SIZE)


def _boundary_token_ids() -> torch.Tensor:
    return torch.tensor(
        [0, SHARD_SIZE - 1, SHARD_SIZE, VOCAB_SIZE - 1],
        dtype=torch.int64,
        device="cuda",
    )


def _make_layer(use_attn_tp_group: bool, rank: int) -> VocabParallelEmbedding:
    layer = VocabParallelEmbedding(
        VOCAB_SIZE,
        HIDDEN_SIZE,
        params_dtype=torch.float32,
        padding_size=1,
        enable_tp=True,
        use_attn_tp_group=use_attn_tp_group,
    ).cuda()
    global_weight = _global_weight()
    if use_attn_tp_group:
        layer.weight.data.copy_(global_weight)
    else:
        layer.weight.data.copy_(
            global_weight[rank * SHARD_SIZE : (rank + 1) * SHARD_SIZE]
        )
    return layer


class TestGlm5NextNextnEmbedding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")

    def test_inherited_attention_tp_layout_gathers_global_ids(self):
        token_ids = _boundary_token_ids()
        reference = F.embedding(token_ids, _global_weight())

        with patch.object(
            vpe, "get_tp_group", return_value=SimpleNamespace(world_size=1)
        ):
            with patch.object(get_flags().dp, "enabled", True):
                self.assertEqual(
                    get_embedding_tp_kwargs(),
                    {"enable_tp": True, "use_attn_tp_group": True},
                )
                self.assertNotIn(
                    "_get_nextn_embedding_tp_kwargs",
                    Glm5NextForConditionalGenerationNextN.__dict__,
                )
                with get_parallel().override(
                    tp_rank=0,
                    tp_size=MODEL_TP_SIZE,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                ):
                    layer = _make_layer(use_attn_tp_group=True, rank=0)
                    self.assertEqual(layer.tp_size, 1)
                    self.assertEqual(layer.shard_indices.org_vocab_start_index, 0)
                    self.assertEqual(
                        layer.shard_indices.org_vocab_end_index, VOCAB_SIZE
                    )
                    with get_forward().scoped(attn_input_scattered=True):
                        output = layer(token_ids)

        self.assertTrue(torch.equal(output, reference))

    def test_model_tp_shard_gathers_reconstruct_global_reference(self):
        token_ids = _boundary_token_ids()
        reference = F.embedding(token_ids, _global_weight())
        outputs = []

        with patch.object(
            vpe, "get_tp_group", return_value=SimpleNamespace(world_size=1)
        ):
            for rank in range(MODEL_TP_SIZE):
                with get_parallel().override(
                    tp_rank=rank,
                    tp_size=MODEL_TP_SIZE,
                    attn_tp_rank=0,
                    attn_tp_size=1,
                ):
                    layer = _make_layer(use_attn_tp_group=False, rank=rank)
                    self.assertEqual(layer.tp_size, MODEL_TP_SIZE)
                    self.assertEqual(
                        layer.shard_indices.org_vocab_start_index,
                        rank * SHARD_SIZE,
                    )
                    self.assertEqual(
                        layer.shard_indices.org_vocab_end_index,
                        (rank + 1) * SHARD_SIZE,
                    )
                    with get_forward().scoped(attn_input_scattered=True):
                        outputs.append(layer(token_ids))

        rank_masks = (
            (token_ids >= 0) & (token_ids < SHARD_SIZE),
            (token_ids >= SHARD_SIZE) & (token_ids < VOCAB_SIZE),
        )
        expected_rank_outputs = tuple(
            reference * rank_mask.unsqueeze(-1) for rank_mask in rank_masks
        )
        for rank, output in enumerate(outputs):
            self.assertTrue(
                torch.equal(output, expected_rank_outputs[rank]),
                f"rank {rank} local shard output mismatch",
            )
        self.assertTrue(torch.equal(outputs[0] + outputs[1], reference))

    def test_out_of_range_diagnostic_is_preserved(self):
        invalid_ids = torch.tensor([VOCAB_SIZE], dtype=torch.int64, device="cuda")

        with patch.object(
            vpe, "get_tp_group", return_value=SimpleNamespace(world_size=1)
        ):
            with get_parallel().override(
                tp_rank=0,
                tp_size=MODEL_TP_SIZE,
                attn_tp_rank=0,
                attn_tp_size=1,
            ):
                layer = _make_layer(use_attn_tp_group=False, rank=0)
                with patch.object(vpe, "maybe_detect_oob") as detect_oob:
                    with get_forward().scoped(attn_input_scattered=True):
                        output = layer(invalid_ids)

        detect_oob.assert_called_once_with(
            invalid_ids,
            0,
            VOCAB_SIZE,
            "VocabParallelEmbedding input id",
        )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))


if __name__ == "__main__":
    unittest.main()
