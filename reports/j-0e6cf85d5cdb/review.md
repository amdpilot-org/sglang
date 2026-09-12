# Independent review of candidate PR 3453

Reviewed exact candidate commit `907fefbb2c13615b1b960f525fa1b451e35f174f`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the
contract in https://github.com/sgl-project/sglang/issues/38074.

## Finding

Recommendation: **accept**. The candidate fully resolves the identified original
defect. On the prepared base, the real Qwen4 loader retained a replaced Parameter
while cyclic GC was disabled, both with and without PLE shards. The retained
Parameter had the loader's FQN-keyed `params_dict` as a direct referrer. At the
exact candidate commit, the same independent cases released the Parameter
immediately.

The cause is the nested `load_qwen4_exp_ple_shard` helper storing warning state on
itself. The function therefore captures itself through its closure, and the same
closure retains `params_dict`. Replacing that function attribute with a nonlocal
boolean breaks the unreachable cycle without changing Marlin repacking or tensor
semantics. Independent repeated-load checks confirmed the warning still fires once
per load and loaded PLE values are unchanged. The candidate regression passed 3
tests / 4 subtests.

The candidate source and regression are byte-for-byte identical to upstream PR
38102 commit `4a8e931a461d0f2c154bd3a2e5c3ba7b34908d69`. The original reporter also recorded
an A/B validation on the issue's exact 2x L20 / TP2 / 48-layer checkpoint: baseline
OOM at layer 9 versus a flat allocation curve through all 48 layers with this fix.
That external evidence supports full resolution, but is not represented as local
execution by this review.

## Environment and scope

Imports resolved to `/job/repo/python/sglang`, including
`python/sglang/srt/models/qwen4_exp.py`. Only Python and report/test files changed;
no C++, CUDA, HIP, FlyDSL, or other native source changed, so no native rebuild was
applicable.

The assigned device is one AMD Instinct MI350X (`gfx950`) with ROCm 7.2. The
original fallback is NVIDIA Marlin on L20 (`sm_89`), so this review could not run
the original CUDA kernel, TP2 deployment, full checkpoint, or directly remeasure
the 0.66 GiB/layer CUDA curve. The lifetime regression uses CPU tensors, although
the ROCm device must remain visible for SGLang imports in this environment.

No remaining counterexample was found within the loader-lifetime contract.
