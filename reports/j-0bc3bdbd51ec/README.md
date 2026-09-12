# Qwen3.8 grounding issue investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/35772

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1296

The prepared `main` snapshot already contains the issue-specific correction.
`Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused` passes the
`MRotaryEmbedding.axis_map` for `[3, T]` positions, and the fused kernel uses
that map to select the temporal, height, or width position independently for
each rotary lane. The reported v0.5.17 behavior instead treated the position
tensor as `[T]`, effectively applying only its temporal row.

The included deterministic GPU probe compares three paths on the assigned
gfx950 GPU:

- current fused `[3, T]` M-RoPE versus the independent native rotary module;
- the legacy temporal-row-only behavior versus that same native reference;
- `[3, T]` positions with equal rows versus ordinary `[T]` RoPE as a boundary
  case.

Observed maximum absolute differences were `0.03125`, `5.875`, and `0.0`,
respectively. The current result is within the BF16 tolerance used by the
checked-in regression; the legacy behavior diverges by two orders of magnitude.
The probe also uses a token-contiguous slice with row stride 68 to cover the
CUDA-graph-buffer layout.

The checked-in kernel unit suite passes all issue-specific numerical cases on
gfx950 after disabling CUDA PDL in the test process. An unmodified run fails at
compilation because `_pdl_supported()` mistakes ROCm capability `(9, 5)` for an
NVIDIA SM value and emits `griddepcontrol.launch_dependents`. This is a separate
ROCm portability issue: the production Qwen3.5 ROCm model path does not select
`forward_prepare_cuda_fused`, and no source workaround was added here.

The Qwen3.8-27B and UI-Mate-27B weights were not present, so no full-model
grounding or Transformers/vLLM comparison was performed. The original issue's
own follow-up reports 20/20 Qwen3.8 cases after applying the same M-RoPE fix,
but that external claim is not represented as a local reproduction.

Run the focused probe with:

```bash
TRITON_CACHE_DIR=/tmp/amdpilot-repo-j-0bc3bdbd51ec/triton-cache-probe \
  /tmp/amdpilot-repo-j-0bc3bdbd51ec/venv/bin/python \
  reports/j-0bc3bdbd51ec/reproduce_mrope_regression.py
```

Raw logs are retained under `raw/`.
