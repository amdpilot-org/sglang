# Independent review of amdpilot-org/sglang PR 2194

Reviewed exact candidate commit: `7bdae6c8600db987d8605c7b7ca8f9a798d16355`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/32521

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2052

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2196

## Recommendation

`request_changes`

The Python change correctly prevents MLX from entering SGLang/Transformers
architecture resolution, and it fixes the reported `auto_map` startup failure
at the model-implementation boundary. It does not establish that Hunyuan can be
served correctly, which is the original issue's broader contract. Upstream PR
https://github.com/sgl-project/sglang/pull/37721 documents an independent
Hunyuan batched-decode correctness blocker: the existing MLX decode wrapper
assumes attention behavior that Hunyuan does not follow. PR 2194 does not alter
that path or add the Hunyuan end-to-end correctness test.

This is therefore a valid partial fix, not a full original-issue fix and not
merely test-only hardening. The candidate prose appropriately limits its claim
to the startup crash, but the review deliverable must judge the original issue.

## Evidence

On the recorded base, an independent fixture exercised the real
`get_resolved_model_impl` -> `get_model_architecture` ->
`resolve_transformers_arch` path. Its Hunyuan-like config had
`architectures=["HunYuanForCausalLM"]` and an `auto_map`; the dynamic class
loader was made to reproduce the reported missing `HunyuanModel` attribute.
The command exited 1 after emitting the remote-module warning and raised:

```
ValueError: Cannot find model module. 'HunYuanForCausalLM' is not a registered
model in the Transformers library and loading the custom model from auto_map
failed.
```

At the exact candidate commit:

- Candidate regression: 3 passed.
- Existing Transformers fallback regression: 2 passed.
- Independent adversarial checks: 6 passed. They verified MLX bypasses a
  failing resolver, does not call that resolver, dominates stale cached
  Transformers state, preserves the non-MLX cache, and preserves non-MLX
  failure behavior.
- Source imports resolved to
  `/job/repo/python/sglang/srt/model_loader/utils.py` and
  `/job/repo/python/sglang/srt/hardware_backend/mlx/runtime.py`.
- The candidate changes only Python and report/test files. No native source or
  generated native library changed, so a native rebuild was not applicable.

Raw logs were retained outside the revision-switching checkout under
`/tmp/amdpilot-repo-j-4ccd342a74b2/evidence/`.

## Architecture and environment limits

The prepared host is Linux x86_64 with one assigned AMD Instinct MI350X
(`gfx950:sramecc+:xnack-`), Torch `2.11.0+rocm7.2`, and HIP `7.2.26015`.
It is not Apple Silicon and has no MLX package, Metal device, or MPS backend.
Enabling `SGLANG_USE_MLX=1` fails the real runtime gate with "MLX is not
installed." The Hunyuan weights were not downloaded. Consequently the original
server command, MLX model loading, token generation, batched decode, and
semantic/numerical equivalence to `mlx_lm` remain unverified. The AMD GPU cannot
qualify an Apple MLX/Metal execution path, so no GPU execution is claimed.
