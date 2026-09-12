# ModelOpt NVFP4 gated MoE TP padding investigation

The prepared main checkout still contained the exact assertion reported by the
source issue. An issue-specific CPU fixture using the Gemma-4 TP=2 dimensions
reproduced both the preceding `(1, 2816, 22)` scale warning and the assertion.

Upstream PR https://github.com/sgl-project/sglang/pull/30888 is open and
unmerged, and contains the same narrow correction. This change adapts that
candidate to the newer prepared checkout and adds malformed-layout boundary
coverage. The implementation pads the two logical `w13` halves independently
before swizzling, together with the corresponding packed `w2` input width and
block-scale width.

Raw failing-before, passing-after, and assigned-GPU tensor check output is in
`reports/j-2708bd88fe6e/raw/`.

The local GPU is one AMD Instinct MI350X/gfx950. It can validate the tensor
transformation but cannot qualify the reported two-GPU NVIDIA Blackwell NVFP4
CUTLASS serving path. Model weights were not available locally.
