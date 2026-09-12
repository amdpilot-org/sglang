# EAGLE radix-cache reservation investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32459

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2073

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, EAGLE still reserved twice
the maximum per-step draft allocation from `kv_committed_len`. Under overlap,
that watermark trails target execution; the extra reserve enters both the
capacity check and actual KV allocation, creating eviction pressure even when
actual KV occupancy is lower. Upstream PR #32574 identifies the same mechanism
but remains open and is not present in this source.

This candidate uses the scheduler's synchronous sequence length (or the exact
host request length when the attention backend intentionally skips the D2H
mirror) and reserves one speculative step. The shared allocation helper accepts
an explicit base length so UNO and DFLASH retain their existing semantics.

The regression was run against a detached worktree at the recorded base: three
new assertions failed there, while all five focused tests pass after the change.
They cover page size 1, page rounding, compatibility with committed-length
callers, encoder image-token accounting, and unchanged UNO reservation.

The assigned MI355X/gfx950 ran an independent numerical GPU check. It cannot
reproduce the report's GLM-5.2 NVFP4 EAGLE setup on eight B200s. The qualified
tiny-Llama fixture from `amdpilot-org/sglang` PR649 was inspected, but it has no
EAGLE draft weights and cannot validate this architecture-specific workload.
Runtime logs are retained under
`/tmp/amdpilot-repo-j-0acafc238b9d/evidence/`; structured claims and exact
commands are in `result.json`.
