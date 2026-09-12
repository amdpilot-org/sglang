# Correction-generation-2 investigation

Candidate: https://github.com/amdpilot-org/sglang/pull/2589 at exact commit `49a136764025c3a0ac76b2105884bbf1215e1e0b`

Independent review: https://github.com/amdpilot-org/sglang/pull/2613

Upstream issue: https://github.com/sgl-project/sglang/issues/30314

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2615

## Result

The candidate's two source corrections are retained unchanged. The candidate regression fails on the recorded base (2 failures: decode lock skipping is default-off, and enabling it reduces the lazy admission ratio from 4 to 3) and passes at the exact candidate (10 tests). The deterministic allocator boundary also confirms the review's specific observation: with `N` request-owned states and `N` admission-locked matched-prefix states in a `2N` pool, one donated-state allocation performs one eviction attempt and then raises `AssertionError: Can not alloc mamba cache`. It does not loop. A `3N` pool succeeds, and `2N` succeeds after the matched-prefix states are evictable.

No additional source correction is justified by the remaining review claims. Turning the allocator assertion into request-level rejection requires a scheduler transaction/rollback design that the available fixture does not validate. The original report and a later independent comment also disagree about `/health` behavior (the original says it times out; the later report says it remains HTTP 200), so a tiny non-Mamba HTTP fixture would not resolve that contract. Direct hierarchical-cache I/O latency, the 222-second TTFT, and the zero-running post-flush state require the missing hybrid-Mamba workload or a faithful deterministic fixture; none is available here.

## Exact limitations

- No Qwen3.5-397B-A17B-FP8 weights or eight H100 80GB GPUs were available.
- The assigned environment is one AMD gfx950 GPU with ROCm 7.2. It cannot qualify TP=8 CUDA, EAGLE, 100K+ traffic, model semantics, or the production direct-I/O setup.
- The qualified tiny Llama fixture has no Mamba state allocator, so it cannot test the condition under review.
- No native source changed; no native rebuild applies.
