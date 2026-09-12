# Correction review: hybrid-mamba NEXTN track indices

Parent candidate: https://github.com/amdpilot-org/sglang/pull/1547 at
`ccabba009ecd391d7f97d50a7c1ca52b7490ac95`.

Independent review: https://github.com/amdpilot-org/sglang/pull/1606.

The candidate's three GPU tests pass at the exact reviewed commit. They correctly
exercise the existing `None -> slot 0` helper behavior, mixed request state, and
the explicit lazy speculative plan. The review is also correct that those tests
alone do not exercise allocation or a boundary crossing.

This correction preserves those tests and adds a lifecycle regression using the
real production methods for fresh-request ping-pong allocation, lazy speculative
preallocation, TARGET_VERIFY metadata construction, and post-crossing promotion
and freeing. The test checks exact GPU-resident slot values (`41 -> 42`) on the
assigned AMD Instinct MI350X (`gfx950`). No runtime source correction was
justified: the prepared base already contains the guard and the complete lazy
speculative lifecycle.

## Evidence

- The historical tensor expression with `None` reproduces the reported
  `TypeError`; run `reproduce_pre_fix.py` with the prepared interpreter.
- Candidate exact commit: 3 helper tests passed on GPU.
- Consolidated correction: 4 tests passed on GPU, including the allocation and
  boundary lifecycle regression.
- Adjacent lifecycle suite: 3 tests and 2 subtests passed.

Raw commands and outputs are retained in `raw/`. Source paths exercised are
`python/sglang/srt/mem_cache/memory_pool.py`,
`python/sglang/srt/managers/schedule_batch.py`,
`python/sglang/srt/speculative/spec_utils.py`, and
`python/sglang/srt/managers/scheduler_components/batch_result_processor.py`.

## Limitations

No fresh-request hybrid-mamba model was served end to end. Qwen3.6-27B-FP8
weights, two RTX 3090 GPUs, NVIDIA execution, and multi-GPU execution were not
available. The repository's existing Qwen3-Next MTP lazy serving test requires a
four-GPU model environment and was not run. The tiny Llama fixture is not a
hybrid-mamba architecture, so using it would not address this issue. This PR
therefore validates transport-free scheduler/cache lifecycle behavior, not model
semantics or serving correctness.
