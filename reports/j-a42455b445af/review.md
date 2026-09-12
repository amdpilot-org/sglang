# Independent review of amdpilot-org/sglang PR 1222

Candidate reviewed: `df73bd27dd394396dd07f8d154f42c9c34006488`

Upstream issue: https://github.com/sgl-project/sglang/issues/36344

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1260

## Recommendation

Accept. The candidate is a source fix, not merely test hardening: it rejects the
reported invalid deterministic Triton configuration during scheduler startup,
before the request can enter the repeat-`OTHER` admission loop. The exact
candidate tests fail on the recorded base and pass at the candidate commit.
Independent boundary tests also pass. Within the original issue's contract,
the change fully resolves the hang by replacing it with a clear configuration
error.

## Evidence

- The prepared checkout exactly matched the recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no prepared/base
  revision difference.
- On the base, the candidate's tests produced `1 failed, 41 passed, 556
  subtests passed`. The failure was the issue-specific assertion that
  deterministic Triton must reject `chunked_prefill_size=128` when alignment
  is 4096. The lower-level regression confirmed two consecutive admissions
  return `OTHER`, do not consume the 128-token budget, and do not make the
  129-token request runnable.
- At exact candidate commit, the focused candidate suite produced `42 passed,
  556 subtests passed`.
- An independent 11-case suite verified rejection for chunks 1, 127, 128, and
  4095 against alignment 4096; acceptance at 4096, 4097, and 8192; disabled
  chunking; acceptance of chunk 128 when the environment alignment is 128; and
  the equivalent FlashInfer guard.
- SGLang and scheduler imports resolved to the checked-out source under
  `/job/repo/python`, not an installed SGLang wheel. Torch resolved to the
  prepared `/opt/venv` ROCm build.
- No native files changed, so no native rebuild was required or performed.

Complete command output captured during revision switching remains outside the
checkout at `/job/review-evidence-j-a42455b445af/`. Concise retained outputs
are also recorded in `raw/test-results.txt`.

## Caveats and limitations

The available GPU is one AMD Instinct MI350X (`gfx950`) with ROCm 7.2, whereas
the reporter used NVIDIA L40S, CUDA, and Triton. No HTTP/model serving run was
used to claim NVIDIA backend equivalence. The issue-specific scheduler state
transition and startup validation are deterministic Python paths and were
tested directly.

An adjacent counterexample remains outside the original report: scheduler
initialization validates the deterministic backend alignment before
`init_dsa_kpool_truncation_align()` can enlarge the final alignment. A
synthetic DSA pool size of 3 changes 4096 to LCM 12288 after a 4096 chunk has
already passed validation. This does not invalidate the reported Triton/L40S
fix, but the final composed alignment should be revalidated in a follow-up if
such non-divisor DSA pool sizes are supported.

The candidate's retained report says `git diff --check` passed, but checking
the complete candidate diff returns 2 because several committed raw log lines
contain trailing whitespace. This is a report-artifact accuracy issue, not a
functional defect in the scheduler change.
