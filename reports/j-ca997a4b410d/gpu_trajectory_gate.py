"""Deterministic one-GPU exercise of the generic trajectory promotion gate."""

import json
import time

import torch

from sglang.multimodal_gen.runtime.utils.compile_trajectory import (
    TrajectoryCapture,
    TrajectoryGate,
    evaluate_trajectory_gate,
)


class StatefulStep(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 32, bias=True)

    def forward(self, state, observation):
        return torch.tanh(self.linear(state) + observation)


def rollout(step, initial, observations):
    capture = TrajectoryCapture()
    state = initial.clone()
    for index, observation in enumerate(observations):
        state = step(state, observation)
        capture.record("state", state)
        capture.decision_trace.append("history:short" if index < 2 else "history:long")
    capture.terminal_state = {"steps": len(observations)}
    return capture


torch.manual_seed(20260912)
device = torch.device("cuda")
model = StatefulStep().to(device).eval()
compiled = torch.compile(model, backend="inductor", fullgraph=True)
initial = torch.randn(4, 32, device=device)
observations = [torch.randn(4, 32, device=device) for _ in range(4)]
gate = TrajectoryGate(
    checkpoints=("state",),
    tensor_thresholds={
        "state": {"max_abs": 2e-5, "mae": 2e-6, "cosine_similarity": 0.99999}
    },
    require_decision_trace_match=True,
)

torch.cuda.synchronize()
start = time.perf_counter()
warm = rollout(compiled, initial, observations)
torch.cuda.synchronize()
cold_seconds = time.perf_counter() - start

reference = rollout(model, initial, observations)
torch.cuda.synchronize()
start = time.perf_counter()
candidate = rollout(compiled, initial, observations)
torch.cuda.synchronize()
warm_seconds = time.perf_counter() - start
result = evaluate_trajectory_gate(reference, candidate, gate)

# A new session starts from the original tensor; captures must not retain the prior request.
reset_candidate = rollout(compiled, initial, observations[:2])
reset_reference = rollout(model, initial, observations[:2])
reset_result = evaluate_trajectory_gate(reset_reference, reset_candidate, gate)

perturbed = rollout(compiled, initial, observations)
perturbed.checkpoints["state"][1][0, 0] += 0.1
perturbation_result = evaluate_trajectory_gate(reference, perturbed, gate)

report = {
    "device": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cold_compile_and_rollout_seconds": cold_seconds,
    "warm_rollout_seconds": warm_seconds,
    "full_trajectory_passed": result.passed,
    "reset_short_history_passed": reset_result.passed,
    "perturbation_rejected": not perturbation_result.passed,
    "perturbation_failures": perturbation_result.failures,
    "per_step_metrics": result.checkpoint_metrics["state"],
}
print(json.dumps(report, indent=2))
assert result.passed
assert reset_result.passed
assert "tensor:state:1:max_abs" in perturbation_result.failures
