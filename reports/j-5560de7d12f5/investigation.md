# MOSS-VL unindexed merger LoRA correction

Upstream issue: https://github.com/sgl-project/sglang/issues/32572

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2384

Candidate parent PR: https://github.com/amdpilot-org/sglang/pull/2274 at exact commit `b53a21a7e9c3ca61f35a09eef878ee8bb9a194d6`.

Independent review parent PR: https://github.com/amdpilot-org/sglang/pull/2348.

## Reproduction

I checked out the exact candidate and independently invoked the real
`LoRAManager.init_lora_modules` method against a model-shaped module tree with
`visual.merger.linear_fc1` and `visual.merger.linear_fc2`. Both selected targets
had no generic layer ID, so neither was wrapped and all four registration
dictionaries remained empty. The retained output is in
`raw/candidate_moss_registration_before.log`.

## Correction

The candidate's valid target normalization, Qwen3-VL deep-stack indexing,
Qwen3-VL dimensions, MOSS-VL dimensions, and path-boundary fixes are preserved.
The correction adds a model-aware LoRA layer resolver and lets MOSS-VL map only
its single unindexed visual merger's `linear_fc1` and `linear_fc2` modules to
logical layer 0. The same resolver is used while wrapping base modules, loading
adapter tensors, and validating UNO targets, so the module and its weights use
the same buffer layer. Generic decoder-layer resolution remains first, and
Qwen3-VL's indexed deep-stack behavior is unchanged.

The regression exercises both module wrapping/registration and adapter tensor
placement. The combined focused suite passes 26 tests; pre-commit passes.

## GPU evidence and limitations

BF16 rank-32 projections for the Qwen3-VL and MOSS-VL merger dimensions ran on
one AMD Instinct MI355X (`gfx950`) and matched independent CPU FP32 references;
see `raw/gpu_linear_fc_reference.log`. This verifies the projection dimensions
and numerical execution only.

Qwen3-VL-8B/EditScore and MOSS-VL model/adapter weights were unavailable, so no
full server/model-load or multimodal semantic run was performed. The available
GPU is ROCm gfx950 rather than the reported H100/CUDA platform. No native source
changed, so a native rebuild was not applicable.
