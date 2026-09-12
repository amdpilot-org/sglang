# Independent review of PR 2082

Reviewed https://github.com/amdpilot-org/sglang/pull/2082 at exact commit
`e0bc6010ab7acc0bfd074aa069b003ee856528b6` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/34112
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2024
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2120

## Recommendation

Accept as a **partial, correctly bounded fix**. The candidate does not fully
verify the original RTX 4090/CUDA and Llama-3.2-1B serving report, and it says
so. Its source changes do independently fix the current-tree form of the
cancelled final-prefill race: the recorded base admits the cancelled request
to optimistic decode and commits a sampled token/metadata; the exact candidate
excludes it while its delayed result is pending and consumes the abort without
committing that result.

The original probe's negative `schedule_batch` / `schedule_batch_post_run`
`output_ids` remain an end-to-end limitation. The downloaded reproducer gets
those fields from an external `backends.sglang.tools.probe_runtime_structures`
package absent from this repository. Current main relays overlap tokens through
`FutureMap.output_tokens_buf`, rather than the older negative handle sequence
described by the historical probe. There is no justified candidate defect or
safe additional source change based only on that unavailable instrumentation.

## Independent evidence

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, with the
  candidate regression methods temporarily overlaid, all three selected cases
  failed: optimistic-decode admission, visible token/metadata commit, and beam
  abort ordering (`3 failed, 17 warnings`).
- On exact candidate commit `e0bc6010ab7acc0bfd074aa069b003ee856528b6`,
  the complete focused suite passed (`23 passed, 17 warnings, 2 subtests`).
- Imports resolved to `/job/repo/python/sglang/...`, confirming the source
  checkout was exercised rather than an installed copy.
- Python compilation and `git diff --check` passed. No native source changed,
  so no native rebuild was applicable.
- One assigned AMD Instinct MI350X (`gfx950`) executed a deterministic ROCm
  tensor operation. This establishes the available architecture only; it does
  not reproduce the reported NVIDIA RTX 4090/CUDA workload.

## Limitations

The exact `meta-llama/Llama-3.2-1B-Instruct` weights, RTX 4090, CUDA runtime,
and external probe instrumentation were unavailable. Consequently the absence
of the original visible-token symptom and negative-ID ladder in that exact
serving workload is unverified. The qualified tiny-Llama fixture cannot prove
model- or architecture-specific equivalence and was not substituted as such.

