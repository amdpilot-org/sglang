# Independent review of amdpilot-org/sglang PR 3195

Candidate: https://github.com/amdpilot-org/sglang/pull/3195

Exact commit: `c3fe7535cc959175a455ba226c11ace9776c6685`

Upstream issue: https://github.com/sgl-project/sglang/issues/35331

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3147

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3198

## Recommendation

`request_changes`. The candidate is a valid partial correction: it fixes both
concrete latent-preparation counterexamples inherited from PR 3139, but it does
not fully implement the served candidate-trajectory contract in the original
issue.

## Verified fixes

- At parent candidate `e3af256016618895e5e59a98a79c454d5ba80bfb`, a list of
  two generators reaches `torch.randn` and raises `TypeError` before denoising.
- At `c3fe7535cc959175a455ba226c11ace9776c6685`, latent preparation uses the
  effective sample batch and `diffusers.utils.torch_utils.randn_tensor`.
- On one AMD Instinct MI350X, a three-candidate latent batch had physical shape
  `(3, 2, 2, 1, 1)`, was bitwise equal to three independent sequential draws
  from seeds 100, 101, and 102, and all candidates were distinct.
- The candidate's focused suite passed: 155 tests, 43 subtests.

## Remaining counterexample

`return_candidates=true` is not honored by the normal served response. The
Cosmos3 decoding stage places `candidate_group` and `candidates` in its internal
payload, but `action_generation_response()` reconstructs a new envelope using
only the reduced `actions` value and drops both fields. The independent case in
`response_contract_counterexample.txt` shows that the default response has
neither candidate identity nor raw candidates. The raw response retains them,
but requiring the non-default raw format is not the advertised request
contract.

This is directly tied to the issue's requirement that the runtime own candidate
identity and final response construction, with optional raw candidates. It also
means the stage-only regression does not prove the HTTP contract.

## Scope and limitations

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` has no
candidate-trajectory module or sampling field, reproducing the original missing
feature. No Cosmos3 model weights were available, so full transformer denoising,
semantic action correctness, and an actual HTTP server request could not be run.
TP/SP reconstruction, CFG-parallel membership, unrelated-request co-batching,
cancellation/failure behavior, action-only video-decode skipping, candidate
metrics, and the requested N={1,2,4,8,10} benchmarks remain unverified or
unimplemented. The candidate itself changes Python only; no native rebuild was
applicable. Imports were confirmed from `/job/repo/python`, with Torch
2.11.0+rocm7.2 and HIP 7.2.26015.

## Commands

Focused candidate tests:

```bash
PYTHONPATH=/job/repo/python /tmp/amdpilot-repo-j-35a48b3bb9fb/venv/bin/python \
  -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_candidate_trajectory.py \
  python/sglang/multimodal_gen/test/unit/test_cosmos3.py \
  python/sglang/multimodal_gen/test/unit/test_pi05_action_api.py
```

The exact standalone reproductions and outputs are retained beside this report.
