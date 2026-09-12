import struct
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.encoders.qwen3vl import Qwen3VLArchConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.quanto_int8_config import (
    QuantoInt8Config,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import (
    gguf_weights_iterator,
    read_gguf_tensor_meta,
)
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import (
    Qwen3VLForConditionalGeneration,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_vision import (
    Qwen3VLVisionRotaryEmbedding,
    Qwen3VLVisionTransformer,
    _vision_cu_seqlens,
    _vision_position_ids,
)
from sglang.srt.models.qwen3_vl import (
    Qwen3VLMoeVisionPatchMerger,
    Qwen3VLVisionPatchEmbed,
)
from sglang.srt.runtime_context import get_parallel


def _write_bf16_gguf(path, name, value):
    """Write one unquantized tensor using GGUF's fastest-axis-first shape."""
    encoded_name = name.encode()
    architecture = b"test"
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    header += struct.pack("<Q", len(b"general.architecture"))
    header += b"general.architecture" + struct.pack("<I", 8)
    header += struct.pack("<Q", len(architecture)) + architecture
    header += struct.pack("<Q", len(encoded_name)) + encoded_name
    header += struct.pack("<I", value.ndim)
    header += b"".join(struct.pack("<Q", dim) for dim in reversed(value.shape))
    header += struct.pack("<IQ", 30, 0)  # GGML_TYPE_BF16, data offset zero.
    header += b"\0" * ((-len(header)) % 32)
    payload = value.contiguous().view(torch.uint16).numpy().tobytes()
    path.write_bytes(header + payload)


class _PatchWeightOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.patch_embed = nn.Module()
        self.model.visual.patch_embed.proj = nn.Conv3d(
            3, 1152, (2, 16, 16), bias=False, dtype=torch.bfloat16
        )

    should_materialize_checkpoint_weight = staticmethod(lambda _name: True)


def _load_patch_weight(model, weights):
    return MiniMaxH3Qwen3VLEncoder.load_weights(model, weights)


def _native_patch_weight():
    values = torch.arange(1152 * 3 * 2 * 16 * 16, dtype=torch.int64)
    return ((values.remainder(251) - 125).float() / 64).to(torch.bfloat16).reshape(
        1152, 3, 2, 16, 16
    )


def test_minimax_h3_loads_folded_bf16_patch_embed_from_real_gguf(tmp_path):
    name = "model.visual.patch_embed.proj.weight"
    native = _native_patch_weight()
    folded = native.reshape(3456, 2, 16, 16)
    checkpoint = tmp_path / "folded.gguf"
    _write_bf16_gguf(checkpoint, name, folded)

    metadata = read_gguf_tensor_meta(str(checkpoint))
    assert metadata[name].logical_shape == (3456, 2, 16, 16)
    model = _PatchWeightOnly()
    loaded = dict(gguf_weights_iterator(str(checkpoint), metadata))[name]
    _load_patch_weight(model, [(name, loaded)])

    actual = model.model.visual.patch_embed.proj.weight
    torch.testing.assert_close(actual, native, rtol=0, atol=0)
    sample = torch.linspace(-1, 1, 3 * 2 * 16 * 16).reshape(1, 3, 2, 16, 16)
    expected = F.conv3d(sample, native.float())
    observed = F.conv3d(sample, actual.float())
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_minimax_h3_patch_embed_load_shapes_and_storage_rules():
    name = "model.visual.patch_embed.proj.weight"
    native = _native_patch_weight()

    for checkpoint_weight in (native, native.transpose(-1, -2)):
        model = _PatchWeightOnly()
        _load_patch_weight(model, [(name, checkpoint_weight)])
        torch.testing.assert_close(
            model.model.visual.patch_embed.proj.weight,
            checkpoint_weight,
            rtol=0,
            atol=0,
        )

    folded_noncontiguous = native.reshape(3456, 2, 16, 16).transpose(-1, -2)
    model = _PatchWeightOnly()
    _load_patch_weight(model, [(name, folded_noncontiguous)])
    torch.testing.assert_close(
        model.model.visual.patch_embed.proj.weight,
        native.transpose(-1, -2).reshape(1152, 3, 2, 16, 16),
        rtol=0,
        atol=0,
    )

    malformed = native.reshape(2304, 3, 16, 16)
    for bad in (malformed, native.reshape(3456, 2, 16, 16).view(torch.uint16)):
        model = _PatchWeightOnly()
        with pytest.raises(RuntimeError, match="Failed to load"):
            _load_patch_weight(model, [(name, bad)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires one GPU")
def test_minimax_h3_folded_patch_embed_gpu_matches_flattened_reference():
    name = "model.visual.patch_embed.proj.weight"
    native = _native_patch_weight()
    model = _PatchWeightOnly()
    _load_patch_weight(model, [(name, native.reshape(3456, 2, 16, 16))])
    model = model.to("cuda")

    sample = torch.linspace(-1, 1, 3 * 2 * 16 * 16, device="cuda").reshape(
        1, 3, 2, 16, 16
    )
    observed = model.model.visual.patch_embed.proj(sample.to(torch.bfloat16))
    # A flattened dot product is independent of the Conv3D implementation and
    # directly tests which checkpoint axes became output and input channels.
    expected = torch.matmul(
        native.float().flatten(1).to("cuda"), sample.float().flatten()
    ).reshape(1, 1152, 1, 1, 1)
    torch.testing.assert_close(observed.float(), expected, rtol=2e-2, atol=0.25)


def test_native_vision_layout_matches_qwen3_merge_order():
    grid_thw = torch.tensor([[1, 4, 6], [2, 2, 4]])

    position_ids = _vision_position_ids(grid_thw, spatial_merge_size=2)
    cu_seqlens = _vision_cu_seqlens(grid_thw)

    assert position_ids.shape == (40, 2)
    assert position_ids[:8].tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
    ]
    assert cu_seqlens.tolist() == [0, 24, 32, 40]


def test_native_vision_keeps_checkpoint_parameter_names():
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="gelu_pytorch_tanh",
        num_heads=2,
        depth=0,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        spatial_merge_size=2,
        out_hidden_size=12,
        deepstack_visual_indexes=[],
    )
    with get_parallel().override(tp_size=1, tp_rank=0):
        model = Qwen3VLVisionTransformer(config)

    assert isinstance(model.patch_embed, Qwen3VLVisionPatchEmbed)
    assert isinstance(model.merger, Qwen3VLMoeVisionPatchMerger)

    assert set(model.state_dict()) == {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "pos_embed.weight",
        "merger.norm.weight",
        "merger.norm.bias",
        "merger.linear_fc1.weight",
        "merger.linear_fc1.bias",
        "merger.linear_fc2.weight",
        "merger.linear_fc2.bias",
    }


def test_native_vision_accepts_srt_linear_quantization():
    config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="gelu_pytorch_tanh",
        num_heads=2,
        depth=1,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        spatial_merge_size=2,
        out_hidden_size=12,
        deepstack_visual_indexes=[],
    )
    prefixes = {
        "model.visual.blocks.0.attn.qkv_proj",
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.0.mlp.linear_fc1",
        "model.visual.blocks.0.mlp.linear_fc2",
    }
    quant_config = QuantoInt8Config(prefixes)
    with get_parallel().override(tp_size=1, tp_rank=0):
        model = Qwen3VLVisionTransformer(
            config,
            quant_config=quant_config,
            prefix="model.visual",
        )

    assert quant_config.selected == prefixes
    for name, parameter in model.blocks[0].named_parameters():
        if name.endswith("weight") and not name.startswith("norm"):
            assert parameter.dtype == torch.int8


def test_native_vision_keeps_position_math_in_fp32():
    class PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1, 8, bias=False, dtype=torch.bfloat16)

        def forward(self, hidden_states):
            return self.proj(hidden_states)

    class BlockRecorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.position_embedding_dtypes = None

        def forward(self, hidden_states, *, position_embeddings, **_kwargs):
            self.position_embedding_dtypes = tuple(
                embedding.dtype for embedding in position_embeddings
            )
            return hidden_states

    class Merger(nn.Module):
        def forward(self, hidden_states):
            return hidden_states.reshape(-1, 4, hidden_states.shape[-1])[:, 0]

    model = Qwen3VLVisionTransformer.__new__(Qwen3VLVisionTransformer)
    nn.Module.__init__(model)
    model.spatial_merge_size = 2
    model.patch_embed = PatchEmbed()
    model.pos_embed = nn.Embedding(16, 8, dtype=torch.bfloat16)
    model.num_grid_per_side = 4
    model.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(2)
    block = BlockRecorder()
    model.blocks = nn.ModuleList([block])
    model.merger = Merger()
    model.deepstack_merger_list = nn.ModuleList()
    model._deepstack_merger_by_layer = {}

    grid_thw = torch.tensor([[1, 4, 6]])
    interpolated_position = model._interpolate_position_embeddings(grid_thw)
    output = model(torch.zeros(24, 1, dtype=torch.bfloat16), grid_thw=grid_thw)

    assert interpolated_position.dtype == torch.float32
    assert output.pooler_output.dtype == torch.bfloat16
    assert block.position_embedding_dtypes == (torch.float32, torch.float32)


def test_qwen3vl_ties_lm_head_to_input_embeddings():
    vision_config = SimpleNamespace(
        hidden_size=16,
        intermediate_size=24,
        hidden_act="gelu_pytorch_tanh",
        num_heads=2,
        depth=0,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        spatial_merge_size=2,
        out_hidden_size=16,
        deepstack_visual_indexes=[],
    )
    text_config = SimpleNamespace(
        hidden_size=16,
        vocab_size=32,
        pad_token_id=0,
        num_hidden_layers=0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    arch_config = SimpleNamespace(
        vision_config=vision_config,
        text_config=text_config,
        tie_word_embeddings=True,
        _fsdp_shard_conditions=[],
        stacked_params_mapping=[],
    )
    config = SimpleNamespace(arch_config=arch_config)

    with get_parallel().override(tp_size=1, tp_rank=0):
        model = Qwen3VLForConditionalGeneration(config)

    assert model.lm_head.weight is model.model.get_input_embeddings().weight
    parameters = dict(model.named_parameters())
    parameters_with_duplicates = dict(model.named_parameters(remove_duplicate=False))
    assert "model.language_model.embed_tokens.weight" in parameters
    assert "lm_head.weight" not in parameters
    assert (
        parameters_with_duplicates["lm_head.weight"]
        is parameters["model.language_model.embed_tokens.weight"]
    )


def test_qwen3_multimodal_encoders_layerwise_offload_vision_blocks():
    assert "model.visual.blocks" in Qwen3VLForConditionalGeneration.layer_names
    assert "model.visual.blocks" in MiniMaxH3Qwen3VLEncoder.layer_names
    assert any(
        condition.__name__ == "is_block"
        for condition in Qwen3VLArchConfig()._fsdp_shard_conditions
    )
