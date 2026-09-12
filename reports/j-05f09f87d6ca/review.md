# Independent review of PR 2005

Upstream issue: https://github.com/sgl-project/sglang/issues/32938

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1975

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2041

Candidate commit: `9bf7abe67090b5c111a7915eb353a8b9cce11dbc`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The candidate correctly changes the configuration-level route implicated by the
issue follow-up, but the available environment cannot independently establish
that it fully resolves the original H200 throughput regression. Recommendation:
**unverified**.

On the recorded base, the reported Hopper + Kimi-K3 + DSPARK + explicit
FlashMLA + `fp8_e4m3` configuration returns no model override, leaving the
default speculative target-verification mode at `prefill`. At the exact
candidate commit, the same configuration returns
`{"speculative_attention_mode": "decode"}`. The candidate's focused regression
passes, and direct source inspection confirms that `FlashMLABackend` handles
target verification and enters its FP8 query-quantization / FP8 KV-cache call
path.

This is a real routing correction, not merely test hardening. It is not an
independent reproduction of the original performance failure or proof of the
reported 3.86x recovery. Those require CUDA FlashMLA on H200, Kimi-K3 and draft
weights, and the TP32/EP32 four-node workload. The assigned machine has one AMD
Instinct MI355X (`gfx950`) and ROCm 7.2. Running an AMD smoke would not exercise
the affected native path, so no GPU execution is claimed.

## Independent boundaries

The candidate leaves BF16 FlashMLA, FA3, and non-Hopper gfx950 on their prior
routes. It routes the reported q_len=8 case and also routes synthetic q_len=1
and q_len=64 cases; the latter widths were not validated against CUDA FlashMLA
and are outside the original reproduction. An explicitly supplied
`speculative_attention_mode=prefill` is overridden to `decode` for the affected
combination, consistent with the existing SM100 Kimi-K3 override behavior but
not covered by the candidate regression.

No source or native C++ changed. A native rebuild was therefore neither needed
nor performed. Imports were verified to resolve to `/job/repo/python/sglang`
for SGLang Python sources. The installed `sgl_kernel.flash_mla` comes from the
prepared `sglang_kernel-0.4.6.post1` egg under `/opt/venv`; on this ROCm host it
cannot validate the CUDA/H200 implementation.

Raw command output was preserved outside revision switching under
`/job/review-evidence/{base,candidate}`.
