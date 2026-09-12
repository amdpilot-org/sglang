# Correction-generation 2 review

Candidate: https://github.com/amdpilot-org/sglang/pull/2608 at exact commit
`8bb43fd1dd0ad1d4910508a98bfa464c86b0835c`.

Independent review: https://github.com/amdpilot-org/sglang/pull/2625.

The candidate's three focused loader regressions pass unchanged. Its useful
test coverage is retained. The concrete remaining counterexample also
reproduces: this prepared environment is reported as AMD Instinct MI350X by
both PyTorch and `rocm-smi`, despite the candidate's correction report naming
MI355X. Both environments report gfx950, so the architecture string alone
must not be used to infer the marketed product name. The earlier report is now
worded as an environment-specific observation rather than a general identity.

The prepared base already contains the production correction that preserves
canonical NPU MoE parameter layout across post-processing and reload. No
additional production source change is justified by the available evidence.

No Ascend device, CANN, or torch_npu runtime is installed, so actual
`npu_format_cast` and Ascend grouped matmul remain unexecuted. The reported
DeepSeekV3.2 checkpoint is unavailable, so no full Engine or HTTP
`update_weights_from_disk` run established scheduler survival. These are
explicit limitations, not evidence for a speculative source change.
