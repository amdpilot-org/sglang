# Independent review of candidate PR 2290

Upstream issue: https://github.com/sgl-project/sglang/issues/31568

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2226

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2324

Candidate: https://github.com/amdpilot-org/sglang/pull/2290 at exact commit
`97894db357f711b28bca4ce7a0f279d2609e806e`.

## Recommendation

Accept. The candidate fully resolves both cache-key-pollution instances in the
original issue. This is a source fix with a failing-before/passing-after
regression, not test-only hardening.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate regression
independently failed both static contract checks: `N` remained a constexpr
parameter of `masked_set_kv_buffer_kernel`, and `NT_BUCKET` remained in the XPU
kernel signature/autotune key/launch. The GPU subtest also used the corrected
post-fix calling convention and consequently failed against the base signature.

At the exact candidate commit, all candidate tests passed. Independent GPU
cases held H=2 and D=65 fixed while varying N across 1, 7, and 65, and exercised
none/all/mixed masks. They matched a PyTorch reference and added exactly one
Triton compiled-cache entry across all three N values. Imports resolved to
`/job/repo/python/sglang/srt/mem_cache/memory_pool.py`, not an installed copy.

The candidate removes `NT_BUCKET` from all three XPU specialization surfaces:
the kernel signature, autotune key, and launch. `NT_BUCKET` was not referenced
by the kernel body, and the autotuner has only one configuration, so its removal
does not remove an effective tuning choice. The related open upstream PR 31689
addresses only the masked KV-write half; it does not invalidate this candidate's
two-part review.

## Environment and limits

- GPU execution used one AMD Instinct MI350X (`gfx950`) with Torch
  `2.11.0+rocm7.2` and HIP `7.2.26015`.
- No native C++ changed, `repository-environment.json` declares no prepared
  native component, and no native rebuild was applicable.
- The XPU FLA kernel was verified by AST/source inspection only because this
  host has AMD gfx950 hardware, not XPU hardware. No XPU numerical or compiler
  execution is claimed.
- An exploratory non-contiguous innermost-D input failed because this kernel's
  API carries B/H strides but no D stride. That is a pre-existing unsupported
  input layout, is outside the original cache-key contract, and is not a
  counterexample to this candidate.

Raw review evidence was preserved outside revision switching at
`/job/review-evidence-j-969a77b43ae2/`.

## Reproduction commands

All commands used `PYTHONPATH=/job/repo/python` and the prepared interpreter
`/tmp/amdpilot-repo-j-969a77b43ae2/venv/bin/python`.

```bash
HIP_VISIBLE_DEVICES=0 python -m pytest -q \
  test/registered/kernels/test_triton_cache_key_hygiene.py -vv

HIP_VISIBLE_DEVICES=0 python \
  /job/review-evidence-j-969a77b43ae2/adversarial_candidate.py
```
