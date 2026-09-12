# Correction review for sglang#31103

Upstream issue: https://github.com/sgl-project/sglang/issues/31103

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2437

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2345

Independent review PR: https://github.com/amdpilot-org/sglang/pull/2397

## Finding

The review counterexample is valid. At exact candidate commit
`12863b77a051754759334e8752894f633b3bc5fd`, a freed request with
`mamba_next_track_idx=None` and stale mapping row `[101, 102]` produced track
slot `101`; the live row correctly produced `202`. The candidate's fallback
therefore avoided the original tensor-construction exception but converted a
skip condition into a stale, valid-looking destination.

The downstream backend separately used an unmasked accepted-step index for the
main recurrent-state scatter. Since that scatter does not consult the track
destination, merely marking the track destination invalid would not protect the
non-track commit.

The correction retains the candidate's safe in-range gather position, replaces
the gathered output with `-1` for every request whose Mamba state has been
cleared, and masks the corresponding accepted-step index to `-1` before all
backend commit paths. Tests cover a live-only boundary, mixed freed/live rows,
an explicit lazy track-position plan, and the non-track commit argument.

## Evidence

- `evidence/exact-candidate-counterexample.txt`: exact candidate, gfx950,
  `actual=[101, 202]`, assertion failure against `[-1, 202]`.
- `evidence/failing-before.txt`: production regressions before the correction,
  three failures and one live-only pass.
- `evidence/corrected-counterexample.txt`: corrected source, gfx950,
  `actual=[-1, 202]`.
- `evidence/passing-after-focused.txt`: corrected focused suite, 18 passed.

## Limitations

The reporter's NVIDIA H20, Qwen3.6-35B-A3B-FP8 weights, EAGLE draft model,
exact launch configuration, and sustained workload were unavailable. This does
not claim a full-model, HTTP-serving, NVIDIA FA3, semantic-accuracy, or soak
reproduction. The deterministic tests qualify the Python request lifecycle and
state-scatter masking on one AMD MI350X/gfx950. No native code changed.
