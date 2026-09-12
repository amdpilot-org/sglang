# Independent review of PR 1472

Reviewed candidate commit `9feec568e30191971b49749cfb479beb46c59b64` against the original issue on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Finding

Recommendation: **accept**. The candidate fully resolves the three reported contracts in the implementation exercised here:

- serial CFG resets once at timestep 0 rather than again on the negative branch;
- each CFG-parallel rank resets its local TeaCache state at the start of every request, including a negative-only rank;
- skip boundaries are doubled only when both CFG branches execute locally, not when CFG parallel assigns one branch per rank.

The recorded base reproduced both lifecycle failures exactly. The exact candidate passed its focused regression suite and independent adversarial state-machine checks. The independent checks covered two serial requests, positive and negative CFG-parallel ranks, GPU-resident stale cache tensors, noninitial timesteps, a non-CFG request while CFG parallel is globally enabled, fractional/negative boundary resolution, and source-path verification.

## Scope and limitations

The imported candidate source was `/job/repo/python/sglang/multimodal_gen/runtime/cache/teacache.py`, with configuration source `/job/repo/python/sglang/multimodal_gen/configs/sample/teacache.py`. No native/C++/HIP source changed, so no native rebuild was applicable.

GPU execution used one assigned AMD Instinct MI350X (`gfx950`, reported capability `(9, 5)`) to create and clear GPU-resident stale TeaCache tensors and perform a numerical tensor check. No Wan model weights were available, so this review does not claim full diffusion generation, semantic/image-quality validation, or a live multi-rank CFG-parallel run. The deterministic state-machine test directly exercises the lifecycle and boundary logic at issue.

The candidate obtains topology from initialized global server arguments. Its regression patches that runtime dependency; production server startup initializes it. This was not found to violate the serving-path contract, but standalone direct callers must likewise initialize or patch server arguments.

Raw evidence is retained in this report directory.
