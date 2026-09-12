# Correction generation 2: cascade attention

Candidate: https://github.com/amdpilot-org/sglang/pull/3224 at exact commit `362ed639d20c2e872dd054f05c609d8e8bb0baad`

Independent review: https://github.com/amdpilot-org/sglang/pull/3276

Upstream issue: https://github.com/sgl-project/sglang/issues/1715

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3280

## Result

The candidate's eager ordinary-decode cascade implementation and regression are preserved. The review counterexamples reproduce by source inspection: CUDA-graph decode constructs and replays the ordinary decode wrappers, while cascade allocation/selection excludes sliding-window and cross-attention dispatch, speculative decode, and dequant-workspace KV caches.

No speculative source expansion was made. The prepared host has one AMD Instinct MI355X with ROCm 7.2, `torch.version.cuda` is null, and `flashinfer` is absent. Consequently this environment cannot validate the production `BatchPrefillWithPagedKVCacheWrapper.forward_return_lse`/merge execution, CUDA-graph capture and replay, or NVIDIA serving correctness/performance. The qualified tiny Llama fixture cannot exercise a CUDA-only FlashInfer backend, so it would not close those gaps.

The exact candidate test reports `6 passed, 1 skipped, 7 subtests passed`; the skipped test is the real NVIDIA FlashInfer integration. Running the same test against base `358c163250ad3b1f62939b01ce1314a0a31a0365` reports `11 failed, 2 passed, 1 skipped`, preserving failing-before evidence for the eager feature. The portable float64 split-softmax comparison executed on the assigned AMD GPU, but it is not evidence for FlashInfer wrapper or kernel compatibility.

## Remaining work requiring a qualified environment

- Design and validate dynamic shared-prefix selection for CUDA-graph replay without capturing stale Python control flow or incompatible wrapper plans.
- Independently establish whether and how cascade should compose with sliding-window attention, encoder-decoder cross attention, speculative decode, and dequant-workspace KV caches.
- Run the gated real-wrapper numerical test on NVIDIA CUDA with FlashInfer installed.
- Run an NVIDIA end-to-end shared-prefix serving workload and benchmark correctness plus latency/throughput.

No native source changed, so no native rebuild was applicable.
