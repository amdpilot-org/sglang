# GLM-5.2 NVFP4 shared-expert fusion investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/29562

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2434

## Evidence

The published `nvidia/GLM-5.2-NVFP4` metadata reports
`GlmMoeDsaForCausalLM`, 256 routed experts, one shared expert, sparse layers
3 through 77, and excludes every sparse layer's `mlp.shared_experts*` from
ModelOpt FP4 while leaving `mlp.experts` quantized. The current fusion gate
accepted that combination. It therefore remapped the unquantized shared
expert into the packed FP4 routed-expert storage, matching the reported
3072-versus-6144 copy failure.

The regression was run before the implementation change:

```text
pytest -q test/registered/unit/models/test_shared_experts_fusion_gates.py \
  -k 'glm52_mixed_precision or glm52_uniform'
F.  [100%]
TypeError: argument of type 'NoneType' is not iterable
1 failed, 1 passed
```

The failure means the mixed-precision GLM-5.2 configuration returned no
fusion-disable reason. The independent uniform-FP4 boundary already passed.

The correction checks each sparse layer and disables fusion only when its
shared expert is excluded from FP4 while its routed experts are not. It then
delegates all other decisions to the existing DeepSeek/GLM DSA gate.

## Limitations

The prepared host has one AMD Instinct MI350X (gfx950), PyTorch
2.11.0+rocm7.2. It does not have eight RTX PRO 6000 SM120 GPUs or the
GLM-5.2-NVFP4 weights, so the original TP=8 server and NVIDIA DSA kernel path
were not executed. Earlier issue evidence also identifies SM120 DSA dependency
support as a separate serving limitation; this change only fixes the reported
weight-loading mismatch.

The complete fusion-gate test file has one unrelated environment-sensitive
failure: `test_expert_parallelism_blocks_fusion_off_rocm` expects EP fusion to
be rejected, but gfx950 is a supported ROCm architecture and the gate returns
no rejection. All four GLM gate tests pass.
