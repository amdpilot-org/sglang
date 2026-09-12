# Bidirectional sliding-window attention investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33603

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1792

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Findings

The prepared checkout retained both source-level causes described by the issue:

1. `TorchNativeAttnBackend._make_sliding_window_mask` always produced a causal
   mask.
2. `forward_extend` discarded `layer.sliding_window_size` when attention was
   non-causal, including `AttentionType.ENCODER_ONLY`.

The direct pre-fix reproduction in `raw/failing-before-mask.txt` shows that a
five-token, radius-one encoder window omitted all four valid future-token
positions. The command exited 1 on the assertion against the independent
bidirectional mask.

Upstream PR https://github.com/sgl-project/sglang/pull/33600 proposed the same
source correction but was closed without merging. The base commit did not
contain its commit `7d60997f227ee6725831fcd4ee00d9b956eba5a5` or an equivalent
implementation.

## Correction and validation

The mask helper now preserves causal behavior by default and constructs
`abs(q_pos - k_pos) <= sliding_window_size` for non-causal attention. Both SDPA
paths pass their causal mode into the helper, and encoder-only extend calls no
longer discard a configured window. Unit tests cover the reported
bidirectional behavior, the forwarding gate, causal masks with a query offset,
and a zero-width bidirectional boundary.

The assigned AMD Instinct MI355X (`gfx950`) also ran SDPA with the corrected
mask. Its output matched a separately computed masked-softmax reference with
maximum absolute error `1.1920928955078125e-07`; see
`raw/gpu-numerical.txt`. This is a kernel-level numerical check, not a claim of
ModernBERT serving parity.

## Limitation

The checkout has no ModernBERT model implementation. The issue itself points
to an implementation on an external development branch, and the reported
`jhu-clsp/ettin-encoder-17m` weights were not prepared for this job. Therefore
the original HTTP embedding comparison against HuggingFace could not be run.
The deterministic tiny Llama fixture was not substituted because it cannot
qualify this encoder architecture or its bidirectional attention semantics.
