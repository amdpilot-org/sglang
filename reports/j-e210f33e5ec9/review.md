# Independent review of PR 810

Candidate: `ffca52cd2b4fd46c6a290434b7c35a34fde1daa4`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fully resolves the original argument-processing defect, within the architecture and model-availability limitations below.

## Finding

The base implementation calls `apply_deepseek_v4_defaults` before `handle_speculative_decoding`. The former accepts only `EAGLE` and `DSPARK`, while the latter is where `NEXTN` is converted to `EAGLE`. A direct reproduction on the recorded base failed with:

```text
AssertionError: Only EAGLE and DSPARK speculative algorithms are supported for DeepseekV4ForCausalLM
```

The exact candidate adds `NEXTN` to the DeepSeek-V4 allowlist and applies the existing `topk == 1` constraint to both `EAGLE` and `NEXTN`. It leaves later canonicalization intact. Independent execution observed `NEXTN -> EAGLE` and `EAGLE -> EAGLE` after the speculative hook.

No remaining counterexample tied to the original contract was found. `NEXTN` and `EAGLE` both reject top-k 2, `DSPARK` remains accepted by the DeepSeek-V4 hook, and `NGRAM` remains rejected.

## Test evidence

- Base reproduction: failed at the original assertion before alias resolution, as expected.
- Candidate regression: 4 tests plus 2 subtests passed.
- Independent adversarial check: passed all acceptance, canonicalization, and rejection cases.
- Full `test/registered/unit/server_args`: 302 passed; 2 unrelated context-parallel tests failed because this ROCm/HIP platform rejects deprecated prefill CP.

The interpreter imported SGLang and the changed hook from `/job/repo/python/sglang`, so the checked-out source was tested rather than an installed SGLang copy. Torch came from the prepared environment and reported ROCm 7.2.

## Native and architecture scope

The candidate changes only Python source, tests, and its prior report artifacts. It changes no C++, CUDA, HIP, or FlyDSL source, so a native rebuild was not applicable.

The assigned host exposes one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`). The original report used two RTX PRO 6000 Blackwell GPUs (`sm_120`) with TP=2 and DeepSeek-V4-Flash weights. Those weights were unavailable and that architecture/topology could not be reproduced. Accordingly, this review verifies the original configuration-ordering contract, not full model loading, generation accuracy, NVIDIA kernels, or TP=2 execution.

## Issue references

Upstream issue: https://github.com/sgl-project/sglang/issues/38236

Mirror issue: https://github.com/amdpilot-org/sglang/issues/855
