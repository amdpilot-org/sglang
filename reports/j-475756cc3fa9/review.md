# Independent review of amdpilot-org/sglang PR #2836

Reviewed exact candidate commit `f7420f4eb0379aba502980c67f10af5316feb33d`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/29864

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2810

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2872

## Verdict

Recommendation: **accept**, narrowly as accurate test-only hardening for the
current successor implementation. The candidate does **not** itself fully
resolve the original compiler issue: it changes no kernel, Triton, or native
code, and its newly registered suite does not launch `store_cache_4d_kernel`.
Its prose and `not_reproduced` result correctly avoid claiming otherwise.

The production kernel and original regression had already been deleted by
upstream commit `4bea51d885538466caef09223e8beb1c307b4489`, before both the recorded
base and candidate. The candidate's only product-tree change registers the
replacement unified-MHA view suite on AMD CI. This is useful regression
coverage for successor KV-write behavior, rather than a compiler fix.

## Evidence

- On the prepared base, the candidate's faithful copy of the historical kernel
  compiled and produced exact output on ROCm 7.2.26015, Triton 3.7.0, and one
  AMD Instinct MI355X (`gfx950`). Therefore no failing-before compiler crash was
  reproducible in this environment.
- The base replacement suite already passed all 13 tests. The candidate also
  passed the same 13 tests; its behavior change is CI collection only.
- At the exact candidate commit, its two historical-kernel cases passed.
- Independent adversarial cases passed with non-contiguous page/token envelope
  strides, page sizes 1/3/8, int32/int64 locations, asymmetric K/V widths, and
  row widths crossing Triton's 128-element block boundary.
- CI collection confirmed both the pre-existing CPU registration and the new
  `stage-b-test-1-gpu-small-amd` registration.
- The upstream issue contains a later report of the original ten-test file
  passing on MI300X/gfx942 with ROCm 7.2 and Triton 3.7. That is relevant
  external evidence, but it is not a failing-before/passing-after result caused
  by this candidate.

Raw logs, fetched issue/PR JSON, the exact diff, source snapshots, and the
independent adversarial script are retained outside the checkout at
`/job/review-evidence-j-475756cc3fa9/`.

## Limitations and remaining counterexamples

- Assigned hardware was MI355X/gfx950, not the originally reported gfx942
  lane. No local gfx942 qualification was possible.
- The original failing Triton/package revision was unavailable, so the
  `TritonAMDGPUCanonicalizePointers` failure could not be reproduced locally
  and no causal compiler fix was demonstrated.
- The candidate's registered test covers the replacement `set_kv_buffer`
  implementation, not the deleted historical kernel. A regression confined to
  `store_cache_4d_kernel` would not be detected by this CI registration.
- No native source changed, so no native rebuild was required or performed.
- This standalone kernel issue did not require model weights, serving, a full
  model, or a multi-node workload; none was tested.
