# Independent review of amdpilot-org/sglang PR 1523

- Candidate: https://github.com/amdpilot-org/sglang/pull/1523
- Exact candidate commit: `dc8a878577b35c207a96d45ad671584be75f925b`
- Upstream issue: https://github.com/sgl-project/sglang/issues/34920
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1562
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **request_changes**
- Fully resolves original: **false**

## Finding

The candidate is a partial fix, not merely test hardening: for the reported normal
DSpark non-compact target-verify shape, it derives committed prefix lengths and a
fixed seven-token verify extent, and the actual DCP planner proceeds past the
original `torch.cumsum(None)` failure. The candidate regression suite passed on
CPU and the assigned AMD Instinct MI350X/gfx950.

However, the fallback for absent `live_seq_lens_cpu` is inconsistent with the
real caller. In `DSparkVerifyExecutor.run_non_compact`, when
`batch.seq_lens_cpu` is absent but `draft_input.nxt_kv_lens_cpu` exists, the
caller temporarily assigns the already-expanded next-KV lengths before
`ForwardBatch.init_new` snapshots `seq_lens_cpu`. The candidate then treats that
expanded host mirror as committed prefix metadata:

```
GPU committed prefix: [3]
ForwardBatch.seq_lens_cpu snapshot: [10]
verify width: [7]
candidate result: total=[10], prefix_gpu=[3], prefix_cpu=[10], extend=[7]
```

Thus the planner receives mutually inconsistent GPU and CPU prefix lengths and
allocates/indexes using a CPU prefix sum of 10 instead of 3. The candidate test
named `test_dcp_target_verify_uses_forward_cpu_mirror_without_live_lengths`
does not reproduce the actual fallback because it supplies an unexpanded host
mirror (`[3]`) rather than the caller-produced expanded snapshot (`[10]`). This
is a remaining counterexample in the same DSpark non-compact target-verify
contract and should be fixed and covered before acceptance.

## Evidence

The prepared checkout was exactly the recorded base before review. The exact PR
head was fetched into a review ref and verified as the requested commit. During
candidate testing, imports resolved to:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/model_executor/runner/eager_runner.py`
- `/job/repo/python/sglang/srt/layers/dcp/planner.py`

On the base, a direct call to the real planner with target-verify-style missing
extend fields failed at `planner.py:64` with `torch.cumsum(None)`. At the exact
candidate, its focused test file passed (`5 passed`). An independent GPU case
with prefixes `[11, 23]` and verify width 7 reached the real planner and produced
a `[48, 1]` DCP KV buffer with prefix sum 34. Index-generation kernels were
replaced by no-op callables only to isolate the reported metadata handoff; the
planner and GPU tensor allocations were real.

No native/C++ files changed in the candidate, so no native rebuild was required.

## Limitations

Only one AMD Instinct MI350X (`gfx950`) GPU was available. The reported system
used Kimi K3, Mooncake PD disaggregation, eight B300 GPUs per side, TP/EP/DCP 8,
and CUDA. Model weights and that distributed topology were unavailable. This
review therefore does not claim a full serving, model-semantic, CUDA/B300,
eight-rank, or multi-node reproduction. The deterministic tests qualify the
specific Python metadata contract and GPU planner allocation path only.
