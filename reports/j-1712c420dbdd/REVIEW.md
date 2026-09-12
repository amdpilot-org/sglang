# Independent review of PR 3069

Upstream issue: https://github.com/sgl-project/sglang/issues/35334

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3093

Candidate: https://github.com/amdpilot-org/sglang/pull/3069 at exact commit `47424480865aedf18f8603ba923cb80e56768237`.

## Recommendation

Request changes. The candidate is a valid bounded correction to the two concrete defects reported by the prior review, but it does not fully resolve the original feature request.

Against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the advisor module is absent and importing it fails. The prepared checkout exactly matched that recorded base.

At the exact candidate commit, I independently confirmed:

- a stale report `driver_version` is rejected by exact environment comparison;
- a CFG degree of two is rejected for a one-branch signature;
- the candidate's 17 focused tests pass; and
- the related compatibility set passes with 216 tests and 67 subtests.

The source import check loaded both `sglang` and `parallel_advisor.py` from `/job/repo/python`, so the tests exercised the checked-out candidate rather than an installed copy. No C/C++/CUDA/HIP/native files differ from the recorded base; a native rebuild was therefore not applicable.

## Remaining counterexamples and missing contract

An independent adversarial case constructs an exact, feasible, quality-passing record with `num_gpus=2` and `tensor_parallel_size=2`. `resolve_calibrated_plan` selects it even though no model/backend capability object participates in startup resolution. Arithmetic product checks cannot establish attention-head divisibility or whether that pipeline/backend supports TP at all.

When calibration is rejected, resolution returns `Resolution(plan=None, source="conservative_fallback", ...)`. It does not route to a compatible conservative pipeline recipe, call existing `auto_tune`, or clearly fail for an unsafe non-fitting model. Thus the requested precedence tiers 3-5 are labels rather than an implemented resolution chain.

The candidate also does not implement the original calibration command, isolated launch/warmup/repeated end-to-end protocol, model quality or trajectory execution, communication instrumentation, or calibrated benchmark matrix. Its report schema can store producer-supplied values, but no code here produces that evidence.

Finally, the candidate changes no distributed group, collective, model wrapping, or FSDP implementation. It therefore provides no model-serving evidence that a selected world-size-one plan avoids sharding wrappers, broadcasts, or unnecessary communicator construction.

## Environment and evidence limits

One AMD Instinct MI350X (`gfx950`) was visible with Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and reported driver `7.1.1.31500000`. The allocation cannot exercise TP=2, CFG=2, multi-rank initialization, or routing between separately launched pools. No model weights were available, and the candidate does not provide a calibration runner, so no semantic quality, trajectory, end-to-end diffusion serving, or independent GPU numerical claim is made. A transport-only tiny Llama fixture would not qualify these diffusion/model-specific requirements and was not used.

Raw commands, outputs, exit statuses, source paths, and environment details are retained in this directory.
