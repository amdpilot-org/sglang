# Correction-generation result

Parent candidate: https://github.com/amdpilot-org/sglang/pull/2700  
Independent review: https://github.com/amdpilot-org/sglang/pull/2732

The exact candidate commit `3a026796b162b43de63bef6ba5e2e609524dc8e7`
was reproduced before consolidation. Its focused Mooncake, NIXL, and shared
disaggregation suite passed (93 tests and 17 subtests). The prepared base still
failed the concrete C2 contract for a legal non-speculative ring of two, while
the candidate maps the odd handoff boundary to one endpoint-local row and sends
no row at an even boundary. See `raw/failing-before-passing-after.log`.

No additional source correction is justified by the review counterexamples.
They are missing end-to-end qualification, not observed candidate failures. In
particular, this prepared host is ROCm/HIP while `_handle_dspark` explicitly
accepts only CUDA or NPU, and no DeepSeek-V4.1 target/draft checkpoints or real
Mooncake/NIXL multi-process topology are present. The generic tiny Llama fixture
cannot validate DeepSeek-V4.1 or DSpark semantics, so it was not substituted for
the requested workload.

A fresh GPU check on the assigned AMD Instinct MI355X copied the 4096-byte
pending FP32 row for `N=101` from a source ring of two to the independently
computed destination row in a ring of eight. It matched a CPU reference exactly
and left the neighboring destination row untouched.

