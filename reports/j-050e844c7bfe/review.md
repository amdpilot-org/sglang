# Independent review of PR 3462

Upstream issue: https://github.com/sgl-project/sglang/issues/9889

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3449

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3464

Candidate: https://github.com/amdpilot-org/sglang/pull/3462 at exact commit
`41b3110c732a8d9d6ed2f492b7c290b05c4a4ae6`.

Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
There was no difference between the requested recorded base and the prepared
checkout.

## Recommendation

Accept. The candidate fully addresses the three call sites in the original
issue snapshot without making unjustified changes. It is a source optimization
for the Phi-4 helper, not merely test hardening. Janus is correctly left alone
because its real input is constructed on CPU, and the stale Gemma operation is
absent from the recorded base/current source.

## Independent findings

- On the base, a queued-work control reproduced the documented GPU behavior:
  GPU `nonzero()` blocked the host for 447.85 ms, whereas fixed-output
  `bucketize` returned in 4.80 ms. On the candidate the same independent probe
  measured 177.20 ms versus 4.06 ms. Absolute sleep duration varied, but the
  synchronization distinction was clear.
- The actual Phi helper constructs `chunk_start_idx` and `arange` without a
  device argument, so its production operation is CPU. The candidate therefore
  should not be described as removing an observed production GPU sync at this
  call site. It removes `nonzero` and an `[x_len, chunks + 1]` interval
  temporary with an equivalent fixed-size mapping.
- The candidate's 19 regression cases passed. An independent exhaustive check
  compared the candidate with the recorded-base implementation over 169,708
  sorted boundary/window cases, including empty inputs, duplicates, missing
  zero boundaries, boundaries at and beyond `x_len`, and multiple left/right
  windows. There were no differences in values, ordering-derived behavior,
  shape, dtype, or device.
- Independent repeated CPU measurements completed for sizes 18 and 64 before
  the prepared runner terminated the longer profiling process. Median legacy
  versus candidate times were 36.81 vs 33.59 microseconds at 18 and 39.85 vs
  36.78 microseconds at 64. The committed candidate profiler was also attempted
  twice and terminated after printing the 18/64 results. Consequently, this
  review does not independently endorse every timing number in the candidate
  prose. This does not create a functional counterexample to the patch.
- Janus dispatch through the real `process_one` method observed `aten::nonzero`
  on a CPU bool tensor of shape `[5]`. Its downstream helper iterates the
  indices in Python, supporting the decision not to replace it for GPU-sync
  reasons.
- `gemma3_mm.py` on the base has no reported `.cpu().nonzero()` operation. No
  model-weight execution can validate a call site that no longer exists.

## Source, native, and architecture checks

The reviewed modules imported from `/job/repo/python/sglang/...`. Torch imported
from `/opt/venv/lib/python3.12/site-packages/torch` and reported
`2.11.0+rocm7.2`, HIP `7.2.26015`. The assigned accelerator reported AMD
Instinct MI350X and `gfx950:sramecc+:xnack-`. The candidate report calls it an
MI355X; this review records the runtime's actual product string. No native code
changed and the prepared environment declares no native rebuild target, so
`native_rebuilt` is false.

No serving/model-weight run was needed: the modified function is a deterministic
CPU tensor helper, Janus placement was exercised directly, and the Gemma call is
absent. This review does not qualify model semantic accuracy, another model
architecture, or a distributed workload.

## Evidence retained outside revision switches

Raw artifacts are under `/tmp/amdpilot-repo-j-050e844c7bfe/`, including
`base-probe.json`, `candidate-probe.json`, `candidate-pytest.log`,
`candidate-profile.log`, `light-profile.log`, `import-paths.txt`, issue/PR JSON,
the candidate patch, and the base call-site inventory.

