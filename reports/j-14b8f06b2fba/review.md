# Independent review of PR 3451

Upstream issue: https://github.com/sgl-project/sglang/issues/32089

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3337

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3454

Candidate: https://github.com/amdpilot-org/sglang/pull/3451 at
`43f6b7197e8540c1d14e1350a0743ab4420004d9`.

## Verdict

Recommendation: **accept**. The candidate is test-only hardening, not a new
production fix. The recorded base already contains the complete production
correction: every FULL/LAST capture row is accounted for regardless of whether
the owning request returns hidden states, FULL capture advances by the forwarded
`extend_input_len`, and the scheduler snapshots those lengths when hidden states
are requested. The candidate's two new unit cases accurately pin the original
mixed-request leak and the cached/chunked plus discarded-row boundary.

The combined base plus candidate fully resolves the reported offset-accounting
contract. The candidate should not be described as independently introducing
the fix; it preserves regression coverage for an existing fix.

## Evidence

- The prepared checkout exactly matched the requested base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`; no image-preparation divergence
  was observed.
- The historical defect was reproduced with the actual production helper from
  the parent of the existing upstream fix, commit
  `1685d29f21279349907d8ebbb417a7c339af07a8`. The production loop skipped the
  non-requesting request, then the helper returned rows 0-2 for the requesting
  three-token request instead of rows 5-7. The assertion failed as expected.
- At the exact candidate commit, the focused test file passed: 9 tests and 2
  subtests. This includes the candidate's exact 5+3 mixed-request regression and
  its independent forwarded-length/discarded-row case.
- An independent ROCm test exercised the checked-out production helper using
  device tensors on one AMD Instinct MI350X (`gfx950`). FULL capture correctly
  advanced across a non-returning 4-row request and a discarded 2-row request,
  returning rows 6-8 to the final request. LAST capture advanced one row per
  request and returned the correct second and third rows.
- Imports resolved to `/job/repo/python/sglang` and the checked-out
  `batch_result_processor.py`, not an installed copy. Torch was
  `2.11.0+rocm7.2`, HIP was `7.2.26015`, and exactly one GPU was visible.
- The candidate changes only Python tests and report artifacts. It changes no
  C++/HIP/FlyDSL/native source, so no native rebuild was required or performed.

Raw command output and the exact pre-fix production-loop source excerpt are
retained in `reports/j-14b8f06b2fba/raw/`.

## Limitations

This review validates the scheduler/processor row-accounting contract and real
GPU tensor slicing. It does not claim a full HTTP/model-serving reproduction,
semantic model accuracy, a different model architecture, pipeline or tensor
parallel multi-GPU behavior, or multi-node behavior. Model weights were not
available or needed for this isolated accounting path. The deterministic tiny
Llama fixture would only add transport/engine coverage and would not qualify
those omitted architectures or distributed cases, so it was not substituted
for the issue-specific checks.

No remaining counterexample was found within the original offset-accounting
contract. End-to-end serving with real weights and distributed execution remain
unverified environment/coverage limitations, not known failures.
