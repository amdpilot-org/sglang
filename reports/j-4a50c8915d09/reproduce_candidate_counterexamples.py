import torch

from sglang.multimodal_gen.runtime.models.dits.flux_2 import (
    Flux2Transformer2DModel,
)
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _load_weights_into_module,
)


class Fixture(torch.nn.Module):
    param_names_mapping = Flux2Transformer2DModel.param_names_mapping

    def __init__(self, device):
        super().__init__()
        self.x_embedder = torch.nn.Module()
        self.x_embedder.weight = torch.nn.Parameter(torch.zeros(3, 2, device=device))
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = torch.nn.Module()
        block.attn.to_q.weight = torch.nn.Parameter(torch.zeros(2, 2, device=device))
        self.transformer_blocks = torch.nn.ModuleList([block])


def install_plain_loader(parameter):
    def loader(param, weight):
        assert param.shape == weight.shape
        param.data.copy_(weight)

    parameter.weight_loader = loader


def run(device):
    module = Fixture(device)
    install_plain_loader(module.x_embedder.weight)
    install_plain_loader(module.transformer_blocks[0].attn.to_q.weight)

    mapped = Flux2Transformer2DModel.param_names_mapping
    print("device", device)
    print("img_in_mapping_present", any("img_in" in key for key in mapped))
    try:
        _load_weights_into_module(
            module, [("img_in.weight", torch.zeros(2, 3, device=device))]
        )
    except AssertionError as exc:
        print("mapped_layout_result", type(exc).__name__, repr(str(exc)))
    else:
        raise AssertionError("mapped layout mismatch unexpectedly loaded")

    expected = torch.ones(2, 2, device=device)
    _load_weights_into_module(
        module, [("transformer_blocks.0.attn.to_q.weight", expected)]
    )
    torch.testing.assert_close(module.transformer_blocks[0].attn.to_q.weight, expected)
    print("bf16_exact_name_identity", True)


run(torch.device("cpu"))
if torch.cuda.is_available():
    print("gpu_name", torch.cuda.get_device_name(0))
    print("gpu_arch", torch.cuda.get_device_properties(0).gcnArchName)
    run(torch.device("cuda:0"))
