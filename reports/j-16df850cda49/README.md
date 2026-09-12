# MXFP4 sharded-state investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34448

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1588

The prepared base still contained the reported defect. In the Triton-kernel branch,
`Mxfp4MoEMethod.process_weights_after_loading` placed swizzled weights on the
quantization helper and deleted `layer.w13_weight` and `layer.w2_weight`. A focused
round-trip regression failed before the correction because `w13_weight` was absent
from `state_dict()`.

Upstream PR https://github.com/sgl-project/sglang/pull/34558 is an open candidate for
this exact issue. This branch applies its issue-specific correction to the prepared
base and adapts the regression to request biases explicitly so it is deterministic
on ROCm as well as CPU-only environments.

After the correction, the existing registered Parameters are rebound to the
postprocessed weight and scale storage. This keeps weight-loader metadata and makes
the sharded loader's copies visible to the runtime wrappers. Sharded saves create
contiguous CPU snapshots because safetensors cannot serialize the non-contiguous
postprocessed views. The four externally referenced Parameters are marked to remain
on device so CPU offload cannot replace storage behind the wrappers.

Raw evidence is retained in `raw/`. The assigned gfx950 test confirms real GPU
storage aliasing plus save/load transport, but it is not a gpt-oss-20b, GH200,
NVIDIA Triton-kernel, TP=2, semantic-accuracy, or multi-node reproduction.
