# Independent review of PR 2152

Upstream issue: https://github.com/sgl-project/sglang/issues/32426

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2091

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2186

Candidate reviewed at exact commit `f31e61cce4b4ebfd83f711defdeb91a546445c77`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

Accept as test-only hardening. The candidate changes no production source. Its four
tests accurately lock down the production correction already present on the recorded
base: compressed-tensors NVFP4 MoE uses `[up; gate]` loading for FlashInfer CUTLASS,
keeps `[gate; up]` for TRT-LLM before that backend's explicit reorder, preserves the
default layout for other compressed-tensors MoE schemes, and forwards the scheme
choice into the common fused-MoE loader.

This candidate does not by itself fully resolve the original issue. The production
fix was already in the prepared base via upstream PR 32430. The candidate adds only
regression coverage and review evidence.

## Independent evidence

- The prepared checkout was exactly the recorded base before review; no difference
  from the image-prepared revision was found.
- SGLang imported from `/job/repo/python/sglang`, including the reviewed
  `compressed_tensors.py` and `compressed_tensors_w4a4_nvfp4_moe.py`, rather than
  from an installed SGLang wheel.
- The exact candidate test passed: 4 passed.
- The same test run against exact upstream-fix parent
  `2cbddb842d67b7d16f04c5a7856a0ff9bddc7767` failed: 4 failed. This directly
  establishes failing-before behavior for the missing property/default/forwarding
  correction. The recorded base itself already contains the correction and passed
  4 tests, so an honest failing run on that base is impossible without mutation.
- An independent adversarial test called the real `FusedMoE._load_w13` path with
  distinct gate (`11, 12`) and up (`31, 32`) sentinels. It verified `[gate; up]`
  when the flag is false and `[up; gate]` when true, showing that the property tested
  by the candidate controls actual loader placement rather than being inert metadata.
- `git diff --check` passed for the candidate diff.

Raw command output was preserved outside the revision-switching checkout under
`/job/review-evidence/j-51ff8188fcfc/`.

## Architecture and environment limitations

The assigned accelerator is AMD Instinct MI350X (`gfx950`) with ROCm 7.2 and Torch
2.11.0+rocm7.2. The affected compressed-tensors NVFP4 implementation explicitly
requires NVIDIA Blackwell (SM100+), while the original report used CUDA 13.0 on an
RTX PRO 6000 Blackwell. Therefore no issue-path GPU kernel, full Ornith model, or
serving-generation reproduction was possible. The affected weights were not
downloaded. A tiny Llama transport fixture would not qualify this different model,
quantization, architecture, or backend and was intentionally not substituted.

No native source changed in the candidate, and the prepared environment declares no
separate native artifact for this checkout, so no native rebuild was applicable.

