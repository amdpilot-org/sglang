from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import GPTQMarlinMoEScheme


def scheme(desc_act):
    return GPTQMarlinMoEScheme(GPTQMarlinConfig(
        weight_bits=4, group_size=128, desc_act=desc_act, is_sym=True,
        lm_head_quantized=False, dynamic={}, full_config={}))


def loader(rank, device):
    obj = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(obj)
    obj.moe_tp_rank = rank
    obj.moe_tp_size = 2
    obj.quant_method = SimpleNamespace()
    obj.scheme = None
    obj.quant_config = None
    obj.use_flashinfer_trtllm_moe = False
    obj.use_triton_kernels = False
    obj.use_padded_loading = False
    obj.use_presharded_weights = False
    obj._has_fused_shared = False
    return obj.to(device)


def check(device):
    for desc_act in (False, True):
        layer = torch.nn.Module().to(device)
        layer.moe_tp_size = 2
        scheme(desc_act).create_weights(layer, 2, 256, 128, torch.bfloat16)
        expected_groups = 2 if desc_act else 1
        assert layer.w13_scales.dtype == torch.bfloat16
        assert layer.w2_scales.dtype == torch.bfloat16
        assert layer.w2_scales.shape == (2, expected_groups, 256)
        assert layer.w2_qzeros.shape == (2, expected_groups, 32)
        assert layer.w2_scales.load_full_w2 is desc_act
        assert layer.w2_qzeros.load_full_w2 is desc_act

    for rank in (0, 1):
        for suffix in ("scales", "qzeros"):
            loaded = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            for full in (False, True):
                rows = 4 if full else 2
                param = torch.nn.Parameter(torch.full((1, rows, 4), -99.0, device=device), requires_grad=False)
                param.quant_method = "group"
                param.is_transposed = True
                param.load_full_w2 = full
                loader(rank, device)._weight_loader_impl(
                    param, loaded, f"experts.0.w2_{suffix}", "w2", 0)
                expected = loaded if full else loaded[rank * 2:(rank + 1) * 2]
                assert torch.equal(param[0].cpu(), expected), (device, rank, suffix, full, param)


print("torch", torch.__version__, "hip", torch.version.hip)
print("sglang source", __import__("sglang").__file__)
print("FusedMoE source", __import__("inspect").getsourcefile(FusedMoE))
check("cpu")
print("cpu adversarial cases passed")
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0))
    check("cuda")
    torch.cuda.synchronize()
    print("gpu adversarial cases passed")
else:
    print("gpu unavailable")
