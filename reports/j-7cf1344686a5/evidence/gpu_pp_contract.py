from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.glm5_next import Glm5NextModel


def make_stage(mhc: bool) -> Glm5NextModel:
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


def run_stage(mhc: bool, tensors: dict[str, torch.Tensor]) -> PPProxyTensors:
    return make_stage(mhc).forward(
        input_ids=torch.empty(0, dtype=torch.long, device="cuda"),
        positions=torch.empty(0, dtype=torch.long, device="cuda"),
        forward_batch=SimpleNamespace(can_run_tbo=False),
        pp_proxy_tensors=PPProxyTensors(tensors),
    )


assert torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0)
arch = torch.cuda.get_device_properties(0).gcnArchName

mhc_hidden = torch.arange(24, dtype=torch.float32, device="cuda").reshape(2, 12)
mhc_output = run_stage(True, {"hidden_states": mhc_hidden})
assert set(mhc_output.tensors) == {"hidden_states"}
torch.testing.assert_close(mhc_output["hidden_states"], mhc_hidden)

plain_hidden = torch.arange(6, dtype=torch.float32, device="cuda").reshape(2, 3)
plain_residual = torch.full_like(plain_hidden, 2)
plain_output = run_stage(
    False, {"hidden_states": plain_hidden, "residual": plain_residual}
)
assert set(plain_output.tensors) == {"hidden_states", "residual"}
torch.testing.assert_close(plain_output["hidden_states"], plain_hidden)
torch.testing.assert_close(plain_output["residual"], plain_residual)

print(f"device={device_name}")
print(f"arch={arch}")
print(f"mhc_keys={sorted(mhc_output.tensors)}")
print(f"mhc_checksum={mhc_output['hidden_states'].sum().item()}")
print(f"non_mhc_keys={sorted(plain_output.tensors)}")
print(f"non_mhc_residual_checksum={plain_output['residual'].sum().item()}")
