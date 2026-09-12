# Independent review of amdpilot-org/sglang PR 2700

Candidate: https://github.com/amdpilot-org/sglang/pull/2700

Exact candidate commit: `3a026796b162b43de63bef6ba5e2e609524dc8e7`

Upstream issue: https://github.com/sgl-project/sglang/issues/38819

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2668

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2711

## Recommendation

`request_changes` for the original issue contract. The candidate is a verified partial fix for the C2 boundary-row transfer defect, but it does not fully implement or validate end-to-end DeepSeek-V4.1 PD + DSpark.

## Findings

1. **The original C2 failure is reproduced on the recorded base.** Using the prepared interpreter and the source module at `/job/repo/python/sglang/srt/disaggregation/utils.py`, base commit `358c163250ad3b1f62939b01ce1314a0a31a0365` rejects the legal non-speculative ring size `R_P=2` with `ValueError: C4 ring_size must be a multiple of 4 and at least 8`. See `raw/base-import-and-counterexample.log`.

2. **The exact candidate fixes the narrow C2 row-index and payload contract.** At commit `3a026796b162b43de63bef6ba5e2e609524dc8e7`, independent cases covered different source/destination request slots, `R_P=2` versus `R_D=8`, odd/even boundaries, wraparound, zero length, and invalid capacities. For `N=101`, source request slot 7 maps to row 14 while destination request slot 11 maps to row 92, matching the issue formula. The candidate's focused suite also passes: 93 tests, 17 subtests. See `raw/candidate-import-counterexamples.log` and `raw/candidate-regression-tests.log`.

3. **The row copy was independently checked on the assigned GPU.** On one AMD Instinct MI350X with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`, rows for layers 2, 8, and 14 were copied using the endpoint-local indices above. Each row contained 1024 FP32 values (4096 bytes); all values matched a CPU reference exactly and a neighboring destination row remained untouched. Total logical odd-boundary payload was 12288 bytes. See `raw/candidate-gpu-independent-row-copy.log`.

4. **The candidate does not satisfy the original end-to-end acceptance criteria.** It adds C2 transfer wiring and tests but does not provide DeepSeek-V4.1 PD + DSpark generation evidence for Mooncake or NIXL, streaming/concurrency, chunked prefill, prefix hits, rejection/commit behavior, cancellation, slot reuse, generation parity, speculative acceptance, or throughput. The candidate's own retained result also acknowledges these omissions.

5. **Important original-contract paths remain unsupported or unverified.** DSpark has an explicit CUDA/NPU-only device guard in `python/sglang/srt/arg_groups/speculative_hook.py`; the prepared architecture is ROCm/MI350X. Decode radix cache remains explicitly incompatible with speculative decoding in `python/sglang/srt/arg_groups/pd_disaggregation_hook.py`. No DeepSeek-V4.1 target/draft weights or suitable transport topology were provided. The tiny Llama fixture cannot qualify this model architecture or DSpark semantics, so it was not substituted as proof.

6. **No native rebuild was applicable.** The candidate changes Python, tests, documentation, and report artifacts only; no C/C++/FlyDSL native source changed. Python compilation of changed runtime paths passed. Source imports resolved to the prepared checkout, not an installed wheel.

## Classification

This is a **partial fix**, not a full original-issue fix, test-only hardening, or an unverified claim. The C2 defect is independently verified, including real GPU numerical behavior. The complete feature request remains unresolved and materially unqualified.
