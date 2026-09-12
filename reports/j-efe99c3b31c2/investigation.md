# Independent review of PR 2080

Upstream issue: https://github.com/sgl-project/sglang/issues/33493

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2114

Candidate: https://github.com/amdpilot-org/sglang/pull/2080 at exact commit
`cb093511f20bcc0a981f3b77ff55bd4d7295fcba`

Parent candidate: https://github.com/amdpilot-org/sglang/pull/1887 at exact commit
`4fc5ad8a7df70cd4eb4306b7d08acf190ce7a6d9`

Parent review: https://github.com/amdpilot-org/sglang/pull/1974

## Recommendation

Accept the candidate as the narrow source-level correction, while retaining an
explicit end-to-end qualification limitation. No source-level counterexample was
found in the reviewed scope. `fully_resolves_original` is recorded as false
because the original DeepSeek-V4-Flash-0731 TP4 HTTP workload could not be run on
the available single GPU without qualified model weights.

This is a review report only. It does not duplicate or modify the candidate patch.

## Independent findings

The prepared checkout was exactly the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, the candidate's real
penalizer lifecycle regression failed: after the second DFLASH decode preparation
for `min_new_tokens=2`, the EOS adjustment remained `-inf` rather than becoming
`0.0`. This reproduces the concrete counterexample from PR 1974 through the real
`BatchedMinNewTokensPenalizer`, including creation, per-step advancement, refresh,
and removal.

The checkout was then detached at exact candidate commit
`cb093511f20bcc0a981f3b77ff55bd4d7295fcba`. The focused candidate regression,
verify-adjustment tests, and penalty library tests all passed (47 tests). An
independent heterogeneous GPU case used requests with minimums 0, 1, and 3. It
observed stop-token blocking states `[false, true, false]` after steps one and two,
then `[false, false, false]` at step three, and verified the same states across both
positions of a two-token DFLASH verify block.

The implementation is appropriately located in
`DFlashDraftInputV2.prepare_for_decode`, which is shared by the DFLASH and DSPARK
spec-v2 paths. The stale overlap field is corrected to
`acc_additive_penalties` both where verify adjustments are applied and where the
DSPARK no-op fast path is selected.

## Import, GPU, and native evidence

The candidate was imported from the checkout, not an installed SGLang wheel:

- `sglang`: `/job/repo/python/sglang/__init__.py`
- `dflash_utils`: `/job/repo/python/sglang/srt/speculative/dflash_utils.py`
- interpreter: `/tmp/amdpilot-repo-j-efe99c3b31c2/venv/bin/python`
- Torch: `/opt/venv/lib/python3.12/site-packages/torch`, version `2.11.0+rocm7.2`
- HIP: `7.2.26015`
- GPU: one AMD Instinct MI355X, `gfx950`

The candidate changes only Python, tests, and reports. It changes no C++, HIP,
CUDA, FlyDSL, headers, or other native source, so a native rebuild was not
applicable. Python compilation of all three changed runtime modules succeeded.

## Preserved raw evidence

Raw logs were preserved outside the checkout while revisions were switched under
`/job/review-evidence-j-efe99c3b31c2/`:

- `base/min_new_tokens_regression.log` — SHA-256
  `bcc39fbceabbf1ed6fa62f95fc19fec1a48d47701dd1c69c4de993bbf5a6a9f1`
- `candidate/focused_tests.log` — SHA-256
  `87a945fe16d19aca8631679a33a09cce8ae0bf3cad3e1273d9638c98c45c3b19`
- `candidate/adversarial_gpu_lifecycle.log` — SHA-256
  `404ad7584dafb7a517a9e3df0252ad5f456ac8089f1d791b9df0d7024a5a3640`
- `candidate/import_environment.log` — SHA-256
  `98256ab6e6b74acede6b7df58fb7ffdb39ac93e43b602d95092823461e1c12e4`

## Limitations

The exact HTTP reproduction requires DeepSeek-V4-Flash-0731, DSPARK, and TP4.
Only one gfx950 GPU was assigned and qualified model weights were unavailable.
The deterministic tiny Llama fixture cannot qualify a different architecture or a
distributed TP4 DSPARK workload, so it was not substituted as proof. The GPU test
qualifies the affected penalty lifecycle and verify tensor operation, not model
semantics or the full serving stack.
