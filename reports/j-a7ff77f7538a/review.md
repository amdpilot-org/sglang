# Independent review of candidate PR 2402

Upstream issue: https://github.com/sgl-project/sglang/issues/31011

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2334

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2439

Candidate: https://github.com/amdpilot-org/sglang/pull/2402 at exact commit `055aba252884d1f8995dce00442ef07d8b9b1154`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Recommendation: **unverified**. The candidate is a plausible, narrow XPU-only workaround, and its mocked regression passes, but this review cannot establish that it fully resolves the original issue because Intel XPU hardware, an XPU PyTorch build, XCCL/oneCCL, and two XPU ranks are absent. `fully_resolves_original` is therefore false.

This is not merely test-only hardening: the candidate changes production dispatch on XPU from integer `all_reduce(MIN)` to gather plus a local minimum. It is also not a verified full fix. The candidate's test mocks the collective and only exercises scalar token tensors; it does not prove that XCCL accepts the allocated stacked output layout or that the affected two-rank serving path works.

## Findings

1. The recorded base contains the reported vulnerable behavior. `Sampler._sync_token_ids_across_tp` unconditionally calls integer `all_reduce(..., ReduceOp.MIN)` whenever grammar synchronization or `SYNC_TOKEN_IDS_ACROSS_TP` activates. A deterministic simulation of the reported XCCL fallback changed `[42, 7]` to `[84, 14]` and failed the expected no-op assertion. This reproduces the application-level consequence against the actual base source, but not the physical XCCL defect.

2. At the exact candidate commit, the candidate regression passes (`4 passed`). It verifies dispatch away from `all_reduce` under a mocked `is_xpu()` and verifies a mocked local minimum for identical values, differing values, and zero.

3. The candidate imports the edited sampler from `/job/repo/python/sglang/srt/layers/sampler.py`. Torch imports from `/opt/venv/lib/python3.12/site-packages/torch/__init__.py` and is `2.11.0+rocm7.2`. There is no XPU runtime. No native source is changed, so no native rebuild is applicable.

4. An independent real collective boundary test found a portability counterexample. With two Gloo ranks, candidate execution fails before reduction because `all_gather_into_tensor` rejects output shape `(world_size, batch)` and expects concatenation along the input's first dimension. This does not prove XCCL fails: SGLang's existing `XpuCommunicator.gather` uses the same stacked layout, and PyTorch backends differ in accepted layout forms. It does prove that the mocked regression is insufficient evidence for the real collective contract.

5. A real single-rank RCCL run on the assigned AMD Instinct MI350X (gfx950) accepted the candidate layout for a three-token int64 tensor and preserved `[42, 7, 0]`. That confirms source import, GPU execution, int64 gather, and local minimum on this ROCm stack only. World size one cannot validate cross-rank synchronization, and RCCL cannot qualify XCCL.

## Remaining counterexamples and required verification

- Run the exact candidate with at least two Intel XPU ranks using the affected XCCL/oneCCL stack and an int64 token batch larger than one. Verify identical IDs stay unchanged and disagreeing ranks resolve elementwise to the minimum.
- Confirm the candidate's `(world_size,) + token_shape` output layout is accepted by the deployed XCCL `all_gather_into_tensor`; Gloo rejects that layout in this environment.
- Exercise the actual SGLang TP grammar/structured-decoding path, or at minimum the real sampler method with XCCL, rather than a mocked collective. No model weights were needed for the source-level checks, but the absence of Intel hardware prevents the issue-defining backend test.

Raw outputs and the reviewed candidate diff are retained under `reports/j-a7ff77f7538a/raw/`.
