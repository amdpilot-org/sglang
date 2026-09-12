# DSV4 decode retraction correction

Upstream issue: https://github.com/sgl-project/sglang/issues/33385

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2087

Candidate: https://github.com/amdpilot-org/sglang/pull/1957 at `8a7f0fe8ed4c15e7700949898b3a77ca09a0b7fc`

Independent review: https://github.com/amdpilot-org/sglang/pull/2051

The candidate correctly contained `NotImplementedError`, but represented it as
host-capacity exhaustion. Its caller consequently aborted the request, and it
still attempted CPU offload when
`disaggregation_decode_enable_offload_kvcache=False`.

This correction keeps the narrow exception containment while separating three
outcomes: backup saved, supported host pool exhausted, and backup unavailable.
The last outcome uses the existing PD rebootstrap path, which recomputes the
request prefix at the prefill worker. A `cpu_tensor` backend also respects the
decode-offload opt-in flag and goes directly to rebootstrap when it is false.

Raw failing-before, candidate-counterexample, passing-after, related-test, and
single-GPU exact-value backup/restore output is retained in `raw/`.

The original tp8/dp8 DeepSeek-V4 speculative HiSparse workload was not run:
weights and the distributed architecture were unavailable. The one assigned
gfx950 validates only supported MHA host-pool backup/restore mechanics, not
DSV4/HiSparse semantics or the reported race timing. Rebootstrap payloads also
do not support multimodal requests, an existing limitation documented in the
source. No native code changed, so no native rebuild was applicable.
