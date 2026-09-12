# Independent review of PR 2126

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2126 at exact commit `27b843774fe9c6d5c6c4a90bc14c0b2e430f13f9`.

Upstream issue: https://github.com/sgl-project/sglang/issues/33454

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2160

## Decision

Recommendation: **accept**. The candidate fully resolves the source-level contract in the original issue: each scheduled DSpark row is limited by both remaining generation budget and the model context length, padded draft positions repeat the final legal position, and clipping forces the compact ragged target layout. I found no remaining counterexample in the available CPU/source-level and single-gfx950 GPU cases.

This is a full fix of the identified planner position-boundary defect, not merely test hardening. The second candidate commit also corrects the concrete prior-review test-registration defect by moving the regression from the invalid `registered/spec` kind to `registered/unit`; the repository validator passes.

## Evidence

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the issue's direct reproduction produced positions `[1048574, ..., 1048579]`, including four positions outside a 1,048,576-row RoPE table.

After temporarily checking out exact candidate `27b843774fe9c6d5c6c4a90bc14c0b2e430f13f9`:

- Imports resolved to `/job/repo/python/sglang` and the reviewed planner source in `/job/repo/python/sglang/srt/speculative/dspark_components/dspark_planner.py`.
- The candidate regression passed: 4 tests.
- The candidate regression plus related scheduler/ragged suites passed: 39 tests and 18 subtests.
- `scripts/lint/check_registered_tests.py` exited 0.
- An independent real-GPU mixed-budget case produced verify lengths `[6, 2, 1, 1, 2]`, maximum position `1048575`, and ragged `qo_indptr` `[0, 6, 8, 9, 10, 12]`.
- Exhausted rows and invalid context configuration were rejected before model execution.
- `git diff --check` passed.

No native source changed, and the prepared environment declares no separate native build target, so no native rebuild was applicable.

## Limitations

The prepared machine has one AMD Instinct MI355X (gfx950) under ROCm 7.2, not two NVIDIA B300 GPUs under CUDA 13. DeepSeek-V4-Flash-0731 weights were unavailable. Therefore this review does not claim a full-model TP2 reproduction of the original CUDA illegal-address crash, semantic-accuracy validation for DeepSeek-V4-Flash, or distributed execution validation. The verified claim is the planner contract that prevents any generated model position from reaching or exceeding `max_position_embeddings` in the exercised boundary paths.
