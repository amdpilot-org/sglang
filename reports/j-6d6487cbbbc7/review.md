# Independent review of PR 2345

Upstream issue: https://github.com/sgl-project/sglang/issues/31103

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2297

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2381

Candidate commit: `12863b77a051754759334e8752894f633b3bc5fd`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Request changes. The candidate is test-only hardening, not a production fix. Its
three GPU cases pass, but they encode an unsafe invariant: every
`mamba_next_track_idx=None` row is expected to select ping-pong position 0.

The prepared base already contains that fallback, so the candidate test passes
unchanged on both base and candidate. The old tensor-construction expression can
reproduce the reported `TypeError`, but that does not prove the fallback is safe
through the real request lifecycle or that the H20/Qwen serving issue is fixed.

## Independent counterexample

The checked-out implementation establishes a second meaning for `None`:
`free_mamba_cache` and `reset_for_retract` clear the request's mamba fields.
Under overlap, a freed request can remain in `TARGET_VERIFY` for one result-lag
iteration. The device ping-pong mapping can still contain the freed slot ids.

On the exact candidate, the adversarial GPU case used a freed row followed by a
live row. The helper returned `[101, 202]`; safe behavior requires `[-1, 202]`
so the freed row is skipped. The positive `101` is stale-but-in-range. Current
`update_mamba_state_after_mtp_verify` also does not mask the corresponding
`last_correct_step_indices`, leaving both track and non-track commit paths able
to write stale recurrent state.

This is consistent with upstream PR #29449, which remains open and explicitly
describes the same crash-to-corruption failure mode. That PR proposes a safe
gather index followed by a `-1` output sentinel and masks the non-track commit.
This review does not apply or endorse that patch; it uses the lifecycle and
downstream code only as adversarial evidence against the candidate's claim.

## Reproduction and paths

Imports resolved to:

- `sglang`: `/job/repo/python/sglang/__init__.py`
- helper: `/job/repo/python/sglang/srt/managers/schedule_batch.py`
- Torch: `2.11.0+rocm7.2`, HIP `7.2.26015`
- GPU: AMD Instinct MI355X, `gfx950:sramecc+:xnack-`

Raw command output was preserved outside revision switches in
`/job/review-evidence-j-6d6487cbbbc7/`. The candidate regression passed 3/3 on
the exact detached commit and when copied onto the recorded base. The independent
freed-row assertion failed with `actual=[101, 202]`.

No native code changed. `repository-environment.json` reports `native: null`, so
no native rebuild was required or performed.

## Architecture and environment limitations

The reporter's NVIDIA H20, Qwen3.6-35B-A3B-FP8 weights, EAGLE draft model,
exact options, and sustained concurrent serving workload were unavailable. A
single AMD gfx950 helper test cannot qualify NVIDIA FA3, full-model execution,
semantic accuracy, HTTP serving, or a soak workload. The tiny Llama fixture is
transport/engine-only and cannot validate this hybrid architecture, so it was
not used as substitute proof.
