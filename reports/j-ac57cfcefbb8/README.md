# ModelOpt expert-wise FP8/W4A16_AWQ investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/36460

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2688

## Result

The requested serving feature could not be implemented and qualified on the
prepared node. The checkout runs ROCm 7.2 on an AMD Instinct MI350X, while
SGLang explicitly rejects `modelopt_fp8`, `modelopt_fp4`, and
`modelopt_mixed` on ROCm. The requested combination also needs a ModelOpt
checkpoint with fused MoE experts exported as `W4A16_AWQ`; no such published
checkpoint was identified. NVIDIA Model Optimizer PR 2017 remains open to
unblock that fused-expert AWQ export path:
https://github.com/NVIDIA/Model-Optimizer/pull/2017

This is not reported as a completed feature. No runtime code was changed.

## Reproduced gap

`ModelOptMixedPrecisionConfig` recognizes the string `W4A16_AWQ` only as an
opaque value. It has no AWQ sub-configuration and no linear or FusedMoE
dispatch branch for that algorithm. More importantly, an expert-wise mixed
map is collapsed at the fused-expert prefix to whichever descendant appears
first. The retained reproduction shows two experts declared as FP8 and
W4A16_AWQ while the fused container resolves wholesale to FP8. Reversing map
insertion order makes it resolve wholesale to W4A16_AWQ, for which
`get_quant_method` returns no MoE method. Either result loses the requested
per-expert contract.

Adding the algorithm name to an existing branch would therefore be unsafe:
ModelOpt INT4 AWQ stores signed INT4 weights packed along output channels,
group scales, and an AWQ `pre_quant_scale`. A correct implementation needs a
heterogeneous expert representation and dispatch that selects FP8 or INT4 per
expert while applying the corresponding expert/projection pre-scale. The
current fused MoE quant-info interfaces carry one weight representation and
one quantization mode for the complete expert set.

## Evidence

- `raw/reproduction.txt`: actual checkout configuration reproduction and
  pinned Torch/ROCm versions.
- `raw/gpu_inventory.txt`: the single assigned AMD Instinct MI350X.
- `raw/modelopt_loader_tests.txt`: the existing ModelOpt loader suite stops at
  SGLang's intentional ROCm rejection before any ModelOpt GPU kernel runs.

## Unverified work

- Loading a real expert-wise FP8/W4A16_AWQ ModelOpt checkpoint.
- Correct heterogeneous expert weight packing and tensor-parallel slicing.
- AWQ `pre_quant_scale` application for each routed expert and projection.
- NVIDIA fused-MoE kernel execution and comparison with an independent BF16
  dequantized reference.
- HTTP serving, accuracy, CUDA graph, tensor parallel, and expert parallel
  behavior for the target architecture.
