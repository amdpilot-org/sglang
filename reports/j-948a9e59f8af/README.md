# Independent review of PR 966

Reviewed candidate commit `b1765e8d5730d1c50116f992d80859ef49da89af`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/37834

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1000

## Finding

The candidate is a narrow source fix, not merely test hardening. It resets
`replayssm_write_pos` when recycled ping-pong slots are initially allocated or
installed as replacements, and admits `extra_buffer` for GDN while retaining
the KDA rejection. The candidate's focused regression fails on the recorded
base (the two cursor assertions fail) and passes at the exact candidate commit.
Independent gfx950 checks also confirm that lazy allocation and device-tensor
replacement reset the selected GPU cursors.

This review does **not** establish the original end-to-end contract. The
prepared environment has one AMD Instinct MI355X (`gfx950`) and no
Qwen/Qwen3.8-27B weights; the report used H200/CUDA. Consequently no actual
hybrid-GDN server run measured repeated-prompt `cached_tokens`, prefix hit
rate, TTFT, or the numerical equivalence of a donated checkpoint containing a
force-flushed ReplaySSM ring. The included unit test mocks the ownership
boundary and does not traverse radix insertion/matching or a decode kernel.

Recommendation: **unverified**. The slot bookkeeping is supported by source
and GPU evidence, but accepting it as a full fix would require an integration
regression showing that a shared prompt reuses both KV and Mamba state with
ReplaySSM plus `extra_buffer`.

## Additional boundary finding

`set_mamba_ping_pong_slot` treats every tensor value as a valid slot. A
tensor-valued `-1` sentinel therefore writes to `replayssm_write_pos[-1]`.
Current in-tree clear callers pass integer `-1`, so this is not evidence that
the reported workload still fails, but the helper's accepted value contract is
not enforced.

## Evidence

Raw command output was retained outside the revision-switched checkout under
`/job/review-evidence-j-948a9e59f8af/`. The source import check resolved
`sglang` and `memory_pool.py` from `/job/repo/python`, not an installed wheel.
No native C++/FlyDSL source changed, so no native rebuild was applicable.

