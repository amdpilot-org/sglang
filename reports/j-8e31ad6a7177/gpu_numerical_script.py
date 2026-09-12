import torch

from sglang.multimodal_gen.runtime.candidate_trajectory import (
    ActionCandidateCapability,
    CandidateTrajectorySpec,
    reduce_action_candidates,
)

assert torch.cuda.is_available()
device = "cuda"
seeds = [1701, 1702, 1703, 1704]
sequential = []
for seed in seeds:
    generator = torch.Generator(device=device).manual_seed(seed)
    sequential.append(torch.randn((1, 32, 14), generator=generator, device=device))
batched = torch.cat(sequential, 0)
repeat = torch.cat(
    [
        torch.randn(
            (1, 32, 14),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
        )
        for seed in seeds
    ],
    0,
)
assert torch.equal(batched, repeat)
capability = ActionCandidateCapability(
    "action_latents",
    "after_denormalization",
    ("mean",),
    "float32",
    "[H,D]",
    ("prompt", "image", "domain"),
    False,
)
output = reduce_action_candidates(
    batched, CandidateTrajectorySpec(4, "mean"), capability
)
reference = torch.stack([item[0].double().cpu() for item in sequential]).mean(0).float()
torch.testing.assert_close(output.cpu(), reference, rtol=1e-6, atol=1e-6)
print("device", torch.cuda.get_device_name(0))
print("torch", torch.__version__, "hip", torch.version.hip)
print("candidate_order_seeds", seeds)
print("batched_equals_independent_seed_draws", torch.equal(batched, repeat))
print(
    "max_abs_vs_float64_cpu_reference",
    (output.cpu() - reference).abs().max().item(),
)
