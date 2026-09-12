# Consolidated correction for PR 3145

Upstream issue: https://github.com/sgl-project/sglang/issues/16255

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3200

Candidate: https://github.com/amdpilot-org/sglang/pull/3145 at exact commit
`8f131570f6d56fa3298720cfec2506660bfd4596`

Independent review: https://github.com/amdpilot-org/sglang/pull/3197

## Result

The review's concrete structural counterexample was reproduced: the candidate's
`deepseek_common/hardware_backend` package contained only the NPU mixin, while
AMD and CPU attention prepare/core routing remained in `deepseek_v2.py`.

This correction preserves the candidate's V3.2 and NPU extractions, adds the
proposed AMD and CPU hardware mixin modules, composes the already-separated
ROCm/CPU attention implementations through those mixins, and removes the AMD
and CPU prepare/core dispatch branches from the model file. The exact candidate
fails the new layout regression; the corrected tree passes it and adversarial
call-signature tests for every moved route.

This is not evidence that every platform conditional in `deepseek_v2.py` can
be moved safely. Remaining branches cover MLP/MoE kernels, quantization,
pipeline embedding policy, stream creation, and capability gates. They were
recorded rather than moved without their unavailable architectures and model
configurations.

## Validation

- Exact candidate: `1 failed, 7 passed`; the dedicated AMD file is absent.
- Corrected focused suite: `30 passed`, `8 subtests passed`.
- Pre-commit: all applicable hooks passed.
- Assigned AMD Instinct MI355X (`gfx950`): the preserved V3.2 empty top-k helper
  matched an independent Torch allocation for device, dtype, and shape.
- No native source changed, so no native rebuild was applicable.

Raw commands and output are retained in `evidence/`.

## Remaining limitations

No DeepSeek V3.2 checkpoint or weights were available, so real Indexer or
IndexerKPool loading, logits, serving, and semantic accuracy remain unverified.
Only one GPU was assigned, so no real pipeline-parallel stage boundary was
crossed. No NPU, NVIDIA, CPU-only, or MUSA runtime was available. AMD and CPU
dispatch is therefore tested at the routing/signature level; the AMD GPU check
only qualifies the stated tensor-allocation invariant. The remaining platform
conditionals are listed in `evidence/structure-before-after.log`; the broader
original refactor remains incremental work.
