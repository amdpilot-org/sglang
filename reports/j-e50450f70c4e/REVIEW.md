# Independent review of PR 1950

Recommendation: **accept** exact commit `bb628ab987cc37b358b1e4f366e5f8b15c19cd83`.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` fails at the same source operation described by the issue: `_process_image_input` unconditionally stacks crop tensors whose tile dimension varies. The independent fixture used three items with 12, 1, and 3 crops and failed before reaching the mocked encoder.

The candidate detects heterogeneous crop shapes, encodes those items separately, and concatenates their feature sequences in input order. It retains the original batched path for equal shapes. This is a production fix, not test-only hardening. It is also consistent with the established per-item implementation in the neighboring DeepSeek-OCR model.

On the exact candidate, all four submitted tests passed. An independent oracle additionally verified a three-item batch, mixed `has_local_crops` flags, spatial metadata, output order and exact values on CPU and the assigned gfx950 GPU. The checked-out `dynamic_preprocess` generated unequal counts for the issue dimensions (24 and 12 on this revision), confirming the fixture targets the real variable-tile contract.

Imports resolved to `/job/repo/python/sglang/srt/models/unlimited_ocr.py` through explicit `PYTHONPATH`; they did not resolve to an unrelated installed SGLang copy. No C/C++/CUDA/HIP/native source changed, so no native rebuild was applicable.

## Scope and limitations

The Unlimited-OCR weights were unavailable. Therefore this review did not run the original HTTP server, validate OCR semantics, or directly observe scheduler survival. The available GPU was an AMD Instinct MI355X (`gfx950`) with torch `2.11.0+rocm7.2`, rather than the reporter's RTX 3090/CUDA environment. These limitations prevent claiming a full model/server reproduction, but do not leave a demonstrated source-level counterexample for valid heterogeneous tile counts.

The candidate does not add general per-request isolation for other multimodal encoder exceptions. That broader scheduler blast-radius concern remains outside this narrow bug fix.

Raw command output and the independent fixture are retained in this directory.
