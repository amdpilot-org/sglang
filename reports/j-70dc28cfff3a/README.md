# BWAP reviewed correction: j-70dc28cfff3a

Outcome: **fixed**

Upstream issue: https://github.com/sgl-project/sglang/issues/35987

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2888

Candidate parent: https://github.com/amdpilot-org/sglang/pull/2760 at
`ee5f9f9b83040ce90406872a74557bc439dfab67`

Independent review parent: https://github.com/amdpilot-org/sglang/pull/2855

The exact candidate reproduced both reviewed defects. It reduced every decode
token to a score and took a running element-wise maximum, whereas Equation 2
requires column L2 pooling over all tokens in each exploration phase before the
per-sample scores enter Equation 3 and activation memory. It also used `round`
instead of the specified floor for top-k width.

The correction preserves the candidate's default-off integration, per-request
schedule, masked fallback, gathered GEMMs, and adaptive graph buffers. It keeps
per-layer, per-request sums of squared row-normalized activations and token
counts, finalizes Equation 2 when each request crosses from exploration to
pruning, then applies the existing element-wise activation-memory maximum. Both
ordinary masks and capture buffers now use floor cardinality.

Evidence:

- `evidence/counterexamples-before.txt`: both failures at the exact candidate.
- `evidence/pytest-after.txt`: 21 tests plus 3 adversarial subtests pass.
- `evidence/gpu-after.txt`: MI350X/gfx950 odd-width GPU comparison matches an
  independent dense masked PyTorch reference exactly.
- `evidence/serving/`: a real graph-enabled server using the qualified random
  tiny-Llama fixture returned HTTP 200 for generate, OpenAI completion, batch,
  and streaming requests. Its log records 2/2 fused layers, graph buffer
  registration, and repeated 64-of-128 adaptive mask refreshes.

The fixture has random weights and qualifies only transport and engine
execution. Production-model semantic accuracy and throughput were not measured
because suitable weights were unavailable. NVIDIA, quantized, biased, TP>1,
speculative-decode, and unsupported architecture paths remain unqualified. No
native C++/HIP/CUDA/FlyDSL source changed, so a native rebuild was not applicable.
