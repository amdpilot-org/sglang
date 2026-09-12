from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.eplb.expert_distribution import (
    _ExpertDistributionRecorderReal,
    _SelectExpertsSinglePassGatherer,
)
from sglang.srt.models import deepseek_v4_dspark
from sglang.srt.utils import Withable

device = torch.device("cuda")
gatherer = _SelectExpertsSinglePassGatherer.__new__(_SelectExpertsSinglePassGatherer)
gatherer._data = torch.zeros((2, 4), dtype=torch.int, device=device)
recorder = _ExpertDistributionRecorderReal.__new__(_ExpertDistributionRecorderReal)
recorder._disable_all = False
recorder._recording = True
recorder._current_layer_idx = Withable()
recorder._current_debug_name = Withable()
recorder._accumulator = SimpleNamespace(
    get_single_pass_gatherer_key=lambda _: "default"
)
recorder._single_pass_gatherers = {"default": gatherer}
topk_ids = torch.tensor([[0, 2], [2, -1]], device=device)


def moe_forward(x, forward_batch, *, input_ids, input_ids_global):
    assert input_ids is None and input_ids_global is None
    recorder.on_select_experts(topk_ids)
    return x + 1


stage = SimpleNamespace(dim=3, _run_moe_ffn_dp_sync=moe_forward)
inputs = torch.arange(12, dtype=torch.float32, device=device).reshape(2, 2, 3)

with recorder.with_current_layer(1):
    recorder.on_select_experts(topk_ids)

graph = torch.cuda.CUDAGraph()
with mock.patch.object(
    deepseek_v4_dspark,
    "get_global_expert_distribution_recorder",
    return_value=recorder,
):
    with torch.cuda.graph(graph):
        output = deepseek_v4_dspark.DSparkV4Stage._run_ffn(stage, inputs, None)

graph.replay()
torch.cuda.synchronize()

with recorder.with_current_layer(1):
    recorder.on_select_experts(topk_ids)

expected_output = inputs + 1
expected_counts = torch.tensor(
    [[0, 0, 0, 0], [2, 0, 4, 0]], dtype=torch.int, device=device
)
torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
torch.testing.assert_close(gatherer._data, expected_counts, rtol=0, atol=0)
assert recorder._disable_all is False

print(f"device={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
print(f"output={output.cpu().tolist()}")
print(f"target_counts={gatherer._data.cpu().tolist()}")
print(f"recorder_disabled={recorder._disable_all}")
