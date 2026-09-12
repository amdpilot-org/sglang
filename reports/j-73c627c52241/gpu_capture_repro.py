import torch

from sglang.srt.eplb.expert_distribution import _ExpertDistributionRecorderReal


class _Gatherer:
    def collect(self):
        return {"value": torch.ones(1, device="cuda")}


class _Accumulator:
    def append(self, _forward_pass_id, _gatherer_key, single_pass_data, _outputs):
        # Matches the issue's unsupported GPU-to-host synchronization at the
        # recorder pass boundary.
        single_pass_data["value"].item()


recorder = _ExpertDistributionRecorderReal.__new__(_ExpertDistributionRecorderReal)
recorder._recording = True
recorder._is_current_stream_capturing = torch.cuda.is_current_stream_capturing
recorder._single_pass_gatherers = {"primary": _Gatherer()}
recorder._accumulator = _Accumulator()

stream = torch.cuda.Stream()
torch.cuda.synchronize()
with torch.cuda.stream(stream):
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        recorder._on_forward_pass_end(1, {})

print("capture_completed")
