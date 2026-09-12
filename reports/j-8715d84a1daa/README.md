# Review correction evidence

This correction was made against prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` and preserves the valid changes from candidate PR https://github.com/amdpilot-org/sglang/pull/1093 at exact commit `a6e24ec6b9964325bc475561c7040235a4cd8f2f`.

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1185

On the exact candidate, the default reserved slot counterexample failed on the assigned gfx950 GPU: a no-rope DCP rank-0 write at virtual location 0 stored a NaN source into physical slot 0. The candidate regression passed because every DCP case disabled reserved-slot handling with `reserved_skip_index=-1`. See `failing-before-default-reserved-slot.log`.

The correction passes `reserved_skip_index` to the no-rope Triton kernel and combines it with the existing DCP owner predicate, matching the rope kernel. The added regression exercises the public default for rope and no-rope paths, checks NaN padding cannot alter slot 0, and checks an owned non-padding location still localizes correctly.

Post-fix logs:

- `passing-after-mla-buffer.log`: 11 passed, 58 skipped. The skips are CUDA-TMA-only cases unavailable on ROCm; the Triton DCP cases executed on GPU.
- `passing-after-capacity.log`: 3 passed, preserving the candidate's index-buffer and DSA capacity fixes.
- `gpu.txt`: assigned GPU identity.

No GLM-5.3-Flash weights or 8xH100 system were available, so the reported full-model watermark workload was not rerun. No source change is inferred from that limitation.
