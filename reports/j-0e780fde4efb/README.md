# Independent review of PR 1957

Upstream issue: https://github.com/sgl-project/sglang/issues/33385

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1999

Candidate: https://github.com/amdpilot-org/sglang/pull/1957 at `8a7f0fe8ed4c15e7700949898b3a77ca09a0b7fc`

## Recommendation

Request changes. The candidate is a useful partial containment: it catches only
`NotImplementedError`, preserves propagation of unrelated failures, and turns
the unsupported backup into the existing per-request abort signal. It therefore
prevents this exception from killing the scheduler.

It does not fully resolve the original contract. An independent `release_req`
probe set decode mode with
`disaggregation_decode_enable_offload_kvcache=False`; the candidate still called
`req.offload_kv_cache()` once. Because DSV4 cannot make that CPU copy, the
candidate freed the device KV, reset the request, returned `False`, and the
caller will abort it. The reported operation is thus converted from a scheduler
crash to request loss, rather than making disabled-offload retraction resumable
or avoiding the unsupported copy.

## Evidence

- `raw/base-reproduction.txt`: failing-before check on recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- `raw/candidate-regression.txt`: candidate's four tests pass at the exact
  reviewed commit.
- `raw/adversarial-release-path.txt`: independent flag-false counterexample and
  release/reset observations.
- `raw/existing-gpu-backup.txt`: two exact-value host-pool backup/restore tests
  pass on the assigned GPU.
- `raw/import-paths-base.txt`: source interpreter/import provenance and ROCm
  version.
- `raw/gpu-arch.txt`: available architecture evidence.
- `raw/base-to-candidate.patch`: review diff preserved outside the temporary
  candidate checkout before returning to the prepared branch.

The imported SGLang source was `/job/repo/python/sglang`; Torch came from the
prepared `/opt/venv` installation and reported `2.11.0+rocm7.2`. The candidate
has no native changes, so a native rebuild was neither required nor performed.

## Limits

The available device was one AMD Instinct MI355X (`gfx950` capability), not the
reported tp8/dp8 deployment. DeepSeek-V4 weights, a multi-node PD setup, and the
reported speculative MTP4/HiSparse load were unavailable. The GPU test covers
supported MHA host-pool transport and restoration only; it does not qualify
DSV4, semantic output, the full serving path, or the original race under load.
