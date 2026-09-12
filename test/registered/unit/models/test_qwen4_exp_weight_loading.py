from types import SimpleNamespace

import pytest
import torch

from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration


PLE_SCALE_NAME = (
    "model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
)


def load_ple_buffer(name, loaded_weight, buffers, loaded_buffers):
    return Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer(
        None, name, loaded_weight, buffers, loaded_buffers
    )


def test_loads_non_unit_ple_embedding_weight_scale():
    """Regression for https://github.com/sgl-project/sglang/issues/36616."""
    destination = torch.ones(1, dtype=torch.bfloat16)
    checkpoint_scale = torch.tensor([0.00019931793212890625])

    model = SimpleNamespace(
        config=SimpleNamespace(
            num_experts=None,
            split_ngram_parts=512,
            tie_word_embeddings=False,
        ),
        language_model_only=False,
        start_layer=0,
        end_layer=2,
        named_parameters=lambda **kwargs: [],
        named_buffers=lambda: [(PLE_SCALE_NAME, destination)],
        named_modules=lambda: [],
        modules=lambda: [],
    )
    model._load_qwen4_exp_ple_buffer = (
        Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer.__get__(model)
    )

    loaded = Qwen4ExpForConditionalGeneration.load_weights(
        model, [(PLE_SCALE_NAME, checkpoint_scale)]
    )

    torch.testing.assert_close(destination, checkpoint_scale.to(torch.bfloat16))
    assert loaded == {PLE_SCALE_NAME}


def test_rejects_mismatched_ple_buffer_shape():
    destination = torch.ones(1, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="Shape mismatch.*expected \\(1,\\).*"):
        load_ple_buffer(
            PLE_SCALE_NAME,
            torch.ones(2),
            {PLE_SCALE_NAME: destination},
            set(),
        )


def test_does_not_claim_unrelated_weight_scale():
    unrelated_name = "model.layers.1.mlp.weight_scale"

    assert not load_ple_buffer(
        unrelated_name,
        torch.tensor([0.25]),
        {unrelated_name: torch.ones(1)},
        set(),
    )
