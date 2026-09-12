# Investigation report: MLA reserved padding slot 0

Upstream issue: https://github.com/sgl-project/sglang/issues/36207

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1219

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

The prepared SGLang source already contains the merged SGLang-owned writer fix
from upstream PR #36003. On the assigned MI350X (`gfx950`), the focused MLA
writer suite passed all five ROCm-applicable tests. Those tests cover the BF16
Triton writer with int32 and int64 locations, the explicit `-1` opt-out, the
FP8-quantized writer, and the scale-buffer writer. The CUDA TMA/JIT tests were
correctly skipped because they require CUDA SM90+.

The issue is nevertheless reproduced in the current fused AITER GLM path. The
deterministic probe initializes cache slot 0 to 17, supplies a NaN padding row
with `slot_mapping=[0, 1]`, and invokes the same
`fused_qk_rope_cat_and_cache_mla` function imported by SGLang. On gfx950, slot
0 changed to 576 NaNs while slot 1 remained finite.

SGLang's call site is
`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py`.
The installed AITER implementation is
`/sgl-workspace/aiter/aiter/ops/triton/fusions/fused_kv_cache.py`; it has no
`pad_slot_id` argument. ROCm/aiter PR #5010 proposes the API-safe opt-in and
kernel predicates, but its GitHub API record was still `open` and unmerged
during this investigation. Passing `pad_slot_id=0` from SGLang before that
dependency is available would break the prepared runtime, so no speculative
call-site change was made.

## Reproduction

```bash
HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-fbe78461ef43/venv/bin/python \
  reports/j-fbe78461ef43/probe_aiter_fused_mla_slot0.py
```

Observed:

```text
device=AMD Instinct MI350X
arch=gfx950:sramecc+:xnack-
slot0_unchanged=False
slot0_nan_count=576
slot1_finite=True
```

Focused existing regression:

```bash
HIP_VISIBLE_DEVICES=0 /tmp/amdpilot-repo-j-fbe78461ef43/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_set_mla_kv_buffer.py
```

Result: `5 passed, 58 skipped`. The skips are CUDA-only TMA coverage and the
expanded CI matrix selected out by the repository's local CI-range helper.

## Evidence and limitations

Raw command output and GitHub API records are under `raw/`. The proposed AITER
PR's API record links its patch, which includes failing-before/passing-after
tests for both fused KV-write branches; it was not applied to the external
AITER checkout.

This did not run the reported GLM-5.2 real weights, TP8/DP4/EP8/MTP workload,
or an eight-GPU serving gate. The single-GPU kernel probe qualifies the exact
writer invariant and valid-slot boundary only; it does not qualify model
semantics or distributed execution. No native SGLang library rebuild was
needed or performed.
