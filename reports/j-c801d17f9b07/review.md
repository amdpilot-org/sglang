# Independent review of PR 1343

Upstream issue: https://github.com/sgl-project/sglang/issues/35705

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1372

Candidate: https://github.com/amdpilot-org/sglang/pull/1343 at `8fe83674054f07a8976e1b2b8b0a465a021ce661`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Accept. The candidate fully resolves the original reported failure within the source-declared contract for `next_token_token_ids_logprobs_val`, whose entries may be either host lists or tensors.

## Evidence

The image-prepared checkout was already on the recorded base; there was no base mismatch. On that base, focused direct calls reproduced `AttributeError: 'list' object has no attribute 'tolist'` in both `move_logprobs_to_cpu` and the duplicated decode normalization path. A tensor-only control passed.

I then detached at the exact candidate commit. Imports resolved to `/job/repo/python/sglang/...`, not an installed SGLang copy. The candidate's four focused regression cases passed. An independent test on one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) mixed real GPU tensors with empty and non-empty host lists in both paths. Results matched a separately constructed NumPy CPU reference, and existing host lists were preserved.

The change contains only Python, tests, and prior report artifacts. No C++, HIP, CUDA, or other native source changed, so a native rebuild was not applicable.

Raw review evidence was intentionally retained outside the checkout at `/job/review-evidence-j-c801d17f9b07/` while revisions were switched. The checkout was returned to `amdpilot/j-c801d17f9b07` before this report was added.

## Scope and limitations

This verifies the original exception and field contract, but not a full serving workload: the original model weights and launch/request configuration were unavailable. The GPU fixture validates scheduler conversion and device-to-host numerical values only. It does not validate model semantics, a different architecture, or distributed execution. MLX-specific selected tests skipped on this ROCm host.

The neighboring `next_token_top_logprobs_val/idx` conversion still has unconditional `.tolist()` calls; upstream PR https://github.com/sgl-project/sglang/pull/35052 addresses those alongside the field from this issue. That adjacent hazard is not a remaining counterexample to the exact `next_token_token_ids_logprobs_val` crash reported here.

## Reproduction commands

Candidate regression:

```bash
PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-c801d17f9b07/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_batch_result_processor_logprob_conversion.py
```

The exact base and independent GPU fixture commands and outputs are summarized in `result.json`; their complete output is retained in the external evidence directory named above.
