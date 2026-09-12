# Independent review of candidate PR 2213

Upstream issue: https://github.com/sgl-project/sglang/issues/32105

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2133

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2248

Candidate: https://github.com/amdpilot-org/sglang/pull/2213 at exact commit `7166628f239de433c463dc11f7870dbd9f5f1ed6`

Recommendation: **accept**. The candidate fully resolves the source-level contract of the original issue. No remaining counterexample was found.

## Evidence

On the required base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a real gfx950 graph capture reached `_ExpertDistributionRecorderReal._on_forward_pass_end`, invoked the fixture accumulator's device `Tensor.item()`, and failed with `hipErrorStreamCaptureUnsupported`. This is the same recorder boundary and forbidden GPU-to-host synchronization described in the original report.

At the exact candidate commit, all four focused unit cases passed. The author's real-GPU reproduction also passed. An independent adversarial capture exercised start boundary, layer hook, and end boundary together: boundary reset/collect/append counters stayed at zero, while the hook's device add was captured and changed the buffer from `0.0` to the independently expected `3.0` after one replay. Thus the candidate avoids both the startup exception and the subtler error of baking boundary bookkeeping into each graph replay without disabling intended hook accumulation.

The candidate changes only Python source and tests. Imports were confirmed from `/job/repo/python/sglang`, not an installed SGLang copy. No native source changed, so no native rebuild was applicable. The candidate is substantively aligned with the open upstream PR 32106.

## Scope and limitations

Review ran on one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) using PyTorch `2.11.0+rocm7.2` and HIP `7.2.26015`. The reported production setup used 8× NVIDIA B200, GLM-5.2 NVFP4, TP8, and built-in MTP. Those weights and hardware were unavailable, so this review does not claim a full server, model-semantic, NVIDIA-specific, or distributed-collective reproduction. The tiny Llama fixture was not used as substitute proof because it cannot qualify the MoE/MTP architecture. These are environment limitations rather than observed counterexamples to the repaired recorder-capture contract.

Raw outputs, import paths, and architecture evidence are retained in this report directory. Candidate and upstream diffs were also preserved outside the checkout while revisions were switched.
