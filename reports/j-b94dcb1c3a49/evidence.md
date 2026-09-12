# Independent review of PR 3418

- Upstream issue: https://github.com/sgl-project/sglang/issues/36829
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3402
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3419
- Candidate: https://github.com/amdpilot-org/sglang/pull/3418
- Exact candidate commit: `dde29de4674681f48047b836068c46dd1215048e`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

Recommendation: **accept as a partial safety fix**. The candidate does not fully
resolve the original issue.

On the recorded base, the candidate regression failed during collection because
`_validate_native_mtp_algorithm` did not exist. At the exact candidate commit,
the focused suite passed 9 tests plus 3 subtests. Independent calls to the actual
validator rejected the same existing local checkpoint expressed exactly, with a
trailing separator, through `..`, through a symlink, and as a relative `./`
alias. It also rejected the same Hub repository with equal revisions and with
omitted target revision versus explicit `main`. A different Hub revision and a
different draft repository remained allowed, as intended by the patch.

The source contract still demonstrates why the rejection is useful:
`Glm5NextForConditionalGeneration.set_eagle3_layers_to_capture` defaults to
three layers, while an mHC state is `hc_mult * H` wide. A small tensor probe on
the assigned gfx950 measured one mHC state as 16384 wide and three concatenated
states as 49152 wide for `H=4096, hc_mult=4`; neither is the bundled NextN
head's expected H-wide input. This is shape evidence only, not execution of the
reported fused CUDA kernel.

An adversarial spelling of two nonexistent relative paths (`missing/glm5` and
`missing/./glm5`) is allowed because both are classified as Hub-like strings.
This is not a reachable checkpoint-loading bypass in the normal hook path:
model configuration is resolved before this validator and those paths do not
identify a loadable checkpoint. Existing relative local aliases are rejected.

## Limits and remaining counterexamples

The machine provides one AMD Instinct MI355X (`gfx950`, ROCm 7.2), not the
reported 4x NVIDIA H20 TP=4 environment. GLM-5.3-Flash weights were unavailable.
Therefore no full server, fused CUDA kernel, semantic acceptance, or throughput
test was possible. In particular:

- NEXTN/EAGLE with the bundled head is permitted but its acceptance and
  throughput are still unverified; the reported acceptance near 1.0 remains
  unresolved.
- A distinct path or revision claimed to contain a trained EAGLE3 draft is
  permitted, but GLM5 architecture compatibility and execution remain
  unverified.
- The candidate prevents the known bundled-head EAGLE3 crash path rather than
  correcting its hidden-state semantics.

No native source changed, so no native rebuild was applicable. Imports resolved
to `/job/repo/python/sglang/srt/arg_groups/speculative_hook.py` and
`/job/repo/python/sglang/srt/models/glm5_next.py`.

## Commands

See `raw/commands.txt` for commands, exit codes, and concise output.
