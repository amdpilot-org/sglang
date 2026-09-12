# Independent review of PR 1330

Reviewed `https://github.com/amdpilot-org/sglang/pull/1330` at exact commit
`51f368efa28a82a10f79f52a57df4e6a192254b4` against upstream issue
`https://github.com/sgl-project/sglang/issues/35771` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/1370`.

## Recommendation

**Unverified.** The source change matches the reported contract: it prevents a
zero-mass candidate from being accepted by either condition and changes the CDF
boundary from inclusive to half-open. I found no source-level counterexample for
valid probabilities and verification coins in `[0, 1)`. However, the changed
native code could neither be rebuilt nor executed on the assigned gfx950 ROCm
host, so the candidate's `candidate_verified` claim is not supported here.

This is not merely test-only hardening: the candidate changes the native
acceptance predicate. It is also not a fully verified original-issue fix in this
review environment. Native execution, DFlash caller-path execution, and
EAGLE/MTP caller-path execution remain unresolved.

## Evidence

- Prepared checkout was clean at the required recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365`; it did not differ from the
  image-prepared revision.
- Python `sglang` imported from `/job/repo/python/sglang`, while `sgl_kernel`
  imported from the preinstalled
  `/opt/venv/lib/python3.12/site-packages/sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg/sgl_kernel`.
- The assigned GPU was an AMD Instinct MI350X, `gfx950:sramecc+:xnack-`, with
  Torch `2.11.0+rocm7.2` and HIP `7.2.26015`.
- On the base, the issue's DFlash reproducer reached GPU tensor setup but raised
  `RuntimeError: DFLASH non-greedy verification is unavailable on this build/device`
  before sampler execution.
- On both base and candidate, the focused AOT tests failed because the installed
  ROCm extension does not register
  `torch.ops.sgl_kernel.tree_speculative_sampling_target_only`.
- At the exact candidate, all seven focused tests collected, including zero
  mass with threshold zero/default, an interior empty CDF bucket, and lower and
  upper RNG endpoints.
- A clean candidate native configure was attempted outside the checkout at
  `/tmp/amdpilot-repo-j-b9e468c18423/sgl-kernel-build`. It failed at
  `project(sgl-kernel LANGUAGES CXX CUDA)` because `nvcc`/CUDAToolkit is absent.
  The available compiler is ROCm HIP clang 22.0.0, but this AOT project has no
  HIP build path.
- The candidate diff itself has whitespace failures inside its committed raw
  evidence logs. This is unrelated to sampler correctness but contradicts an
  unqualified clean-patch claim.

Raw command output was preserved outside the revision-switching checkout under
`/job/review-evidence-j-b9e468c18423/`.

No model weights, full model, semantic-accuracy workload, or distributed
workload were used or claimed.
