# Investigation of sglang#31103

Upstream issue: https://github.com/sgl-project/sglang/issues/31103

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2297

## Finding

The prepared `main` checkout already contains the narrow guard needed for the
reported crash. `set_mamba_track_indices_from_reqs` converts an uninitialized
`req.kv.mamba_next_track_idx` from `None` to the first ping-pong position (`0`)
before constructing the pinned `int64` tensor. Its comment explicitly identifies
the speculative-v2 verify path named in the report.

The older expression was reproduced independently on the assigned gfx950 GPU:
constructing an `int64` tensor from `[0, None, 1]` raises
`TypeError: 'NoneType' object cannot be interpreted as an integer`. The added
regression invokes the actual checked-out helper and verifies:

- a single uninitialized request selects its first ping-pong slot;
- mixed initialized positions (`0`, `1`) and an uninitialized position gather
  the correct per-request slots;
- an explicit speculative track plan overrides uninitialized request state.

Raw output is retained in `evidence/historical-expression.txt` and
`evidence/regression-gpu.txt`.

## Related changes inspected

The earlier report, https://github.com/sgl-project/sglang/issues/29516, contains
the same Mamba/EAGLE `NoneType` traceback and identifies upstream PR #29449 as a
candidate at that time. That PR remains open and was not copied: current `main`
has since substantially refactored this path and contains its own explicit
normalization in `python/sglang/srt/managers/schedule_batch.py` plus a dedicated
pre-verify rebuild in `python/sglang/srt/speculative/spec_utils.py`.

## Limitations

The reporter's NVIDIA H20 hardware, Qwen3.6-35B-A3B-FP8 weights, EAGLE draft
model, launch configuration, concurrency pattern, and long-duration workload
were unavailable. Therefore this does not claim a full-model, NVIDIA, HTTP, or
soak reproduction. The GPU test qualifies only the reported scheduler-helper
failure mechanism and slot-selection behavior on one AMD MI350X/gfx950 GPU.

