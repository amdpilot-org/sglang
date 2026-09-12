from pathlib import Path
import tempfile

import torch
from safetensors.torch import save_file

import sglang.multimodal_gen.runtime.models.parameter as parameter_module
from sglang.multimodal_gen.runtime.models.parameter import ModelWeightParameter
from sglang.multimodal_gen.runtime.post_training.weights_updater import (
    _get_weights_iter,
    _load_weights_into_module,
    compare_module_weights_with_disk,
)


class TwoWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.present = torch.nn.Parameter(torch.tensor([1.0]))
        self.omitted = torch.nn.Parameter(torch.tensor([999.0]))


class ConcreteParallel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        def loader(param, value):
            param.load_column_parallel_weight(value)

        self.weight = ModelWeightParameter(
            output_dim=0,
            input_dim=1,
            data=torch.zeros((2, 2), device=device),
            weight_loader=loader,
        )


print("torch", torch.__version__, torch.__file__)
print("device", torch.cuda.get_device_name(0))
# Single-rank numerical exercise without starting a distributed server.
parameter_module.get_tp_rank = lambda: 0
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    partial = root / "partial"
    partial.mkdir()
    save_file({"present": torch.tensor([1.0])}, partial / "model.safetensors")
    module = TwoWeights().cuda()
    partial_result = compare_module_weights_with_disk(module, str(partial))
    print("PARTIAL_CHECKPOINT", partial_result)
    print("OMITTED_LIVE_VALUE", module.omitted.item())
    assert partial_result["match"] is True
    assert partial_result["parameter_count"] == 1

    parallel = root / "parallel"
    parallel.mkdir()
    disk = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file({"weight": disk}, parallel / "model.safetensors")
    concrete = ConcreteParallel("cuda")
    _load_weights_into_module(concrete, _get_weights_iter(str(parallel)))
    match = compare_module_weights_with_disk(concrete, str(parallel))
    cpu_equal = torch.equal(concrete.weight.detach().cpu(), disk)
    print("CONCRETE_SUBCLASS_MATCH", match, "CPU_REFERENCE", cpu_equal)
    assert match["match"] and cpu_equal
    concrete.weight.data[0, 0] = -1
    mismatch = compare_module_weights_with_disk(concrete, str(parallel))
    print("CONCRETE_SUBCLASS_MISMATCH", mismatch)
    assert mismatch["match"] is False
