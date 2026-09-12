# Investigation: PP + DSA CUDA-graph proxy sizing

- Upstream issue: https://github.com/sgl-project/sglang/issues/38707
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/742
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The reported eight-GPU `gfx942`/GLM-5.3 workload cannot be reproduced on the
assigned single `gfx950` GPU, and the model weights are not present. The source
nevertheless contains an issue-specific shape error in the CUDA-graph PP path:
when `require_attn_tp_gather` is true, the PP boundary is SCATTERED and each TP
rank owns `num_tokens / attn_tp_size` rows, but capture and replay exposed
`num_tokens` rows from the proxy buffers. The same mismatch existed in the
dummy graph warmup path.

This is the invariant described by the still-open related upstream PR
https://github.com/sgl-project/sglang/pull/31012. The prepared base did not
contain that correction. This change centralizes the row-count calculation,
uses it for graph capture input, graph replay output, and dummy warmup, and
rejects invalid non-divisible shapes rather than silently truncating them.

Because the original distributed model/hardware combination is unavailable,
the result is a verified candidate fix, not a claim that the full reported
serving failure was reproduced or eliminated.

## Evidence

- `raw/failing_before_pp_proxy_token_count.txt`: regression fails to import the
  missing rank-local row-count behavior before the implementation.
- `raw/unit_after.txt`: focused regression and neighboring prefill helper tests.
- `raw/decode_graph_suite_final.txt`: 57 decode graph/registry tests pass.
- `raw/gpu_pp_proxy_boundary.txt`: actual ROCm execution on the assigned MI350X
  (`gfx950`) matches an independent NumPy reference for a 32-row, TP=4 proxy:
  8 rank-local rows and sum 1540.

## Limitations

- Only one `gfx950` GPU was assigned; PP=2/4 and TP=4 require multiple GPUs.
- The report targets `gfx942`, which was not available.
- GLM-5.3 753B FP8 weights were not available.
- The tiny Llama fixture cannot qualify DSA, GLM architecture behavior, or a
  distributed PP workload, so it was not used as a substitute reproduction.
- No native code changed and no native rebuild was applicable.

