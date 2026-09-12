import torch

from sglang.multimodal_gen.runtime.models.dits.flux_2 import (
    Flux2Transformer2DModel,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _load_weights_into_module,
)


class _Flux2ReloadFixture(torch.nn.Module):
    """Minimal module exposing the real FLUX.2 checkpoint-name mapping."""

    param_names_mapping = Flux2Transformer2DModel.param_names_mapping

    def __init__(self):
        super().__init__()
        attention = torch.nn.Module()
        attention.to_qkv = torch.nn.Module()
        attention.to_qkv.weight = torch.nn.Parameter(torch.zeros(6, 2))
        attention.to_added_qkv = torch.nn.Module()
        attention.to_added_qkv.weight = torch.nn.Parameter(torch.zeros(6, 2))
        block = torch.nn.Module()
        block.attn = attention
        self.transformer_blocks = torch.nn.ModuleList([block])


def _install_sharded_loader(parameter: torch.nn.Parameter) -> None:
    def loader(param, weight, shard_id):
        param.data[shard_id * 2 : (shard_id + 1) * 2].copy_(weight)

    parameter.weight_loader = loader


def test_flux2_diffusers_qkv_reload_routes_each_fused_shard():
    module = _Flux2ReloadFixture()
    _install_sharded_loader(module.transformer_blocks[0].attn.to_qkv.weight)
    _install_sharded_loader(module.transformer_blocks[0].attn.to_added_qkv.weight)

    prefix = "transformer_blocks.0.attn"
    names = ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj")
    weights = [
        (f"{prefix}.{name}.weight", torch.full((2, 2), value))
        for value, name in enumerate(names, start=1)
    ]

    _load_weights_into_module(module, weights)

    expected_image = torch.tensor([[1.0] * 2] * 2 + [[2.0] * 2] * 2 + [[3.0] * 2] * 2)
    expected_text = torch.tensor([[4.0] * 2] * 2 + [[5.0] * 2] * 2 + [[6.0] * 2] * 2)
    torch.testing.assert_close(
        module.transformer_blocks[0].attn.to_qkv.weight, expected_image
    )
    torch.testing.assert_close(
        module.transformer_blocks[0].attn.to_added_qkv.weight, expected_text
    )


def test_flux2_runtime_layout_reload_keeps_plain_two_argument_loader():
    module = _Flux2ReloadFixture()
    parameter = module.transformer_blocks[0].attn.to_qkv.weight
    calls = []

    def loader(param, weight, *args):
        calls.append(args)
        param.data.copy_(weight)

    parameter.weight_loader = loader
    expected = torch.arange(12, dtype=torch.float32).reshape(6, 2)

    _load_weights_into_module(
        module, [("transformer_blocks.0.attn.to_qkv.weight", expected)]
    )

    assert calls == [()]
    torch.testing.assert_close(parameter, expected)
