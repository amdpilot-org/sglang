# Independent review of PR 666 at 997071d

Upstream issue: https://github.com/sgl-project/sglang/issues/38709

Mirror issue: https://github.com/amdpilot-org/sglang/issues/669

Candidate: https://github.com/amdpilot-org/sglang/pull/666 at `997071da4e91329b0f518b8b2374531348ee6cd6`

## Verdict

Recommendation: **accept**. The candidate fully resolves the original return-contract defect at the implementation boundary. It is not merely test hardening: production code now requests LSE and rejects a tensor-only or otherwise missing-LSE response before collective use.

The full 8-rank deployment was not available. This verdict covers the defect described by the issue, supported by failing-before/passing-after implementation evidence and single-GPU numerical kernel checks; it does not claim an 8x MI300X GLM-5.3 deployment passed.

## Evidence

The prepared base was `358c163250ad3b1f62939b01ce1314a0a31a0365`. Calling the actual `DeepseekMLARocmForwardMixin.forward_absorb_rocm_core` DCP branch with tensor-only backend returns produced:

- batch 1: `ValueError: not enough values to unpack (expected 2, got 1)`;
- batch 2: Python silently unpacked the two output rows as `attn_output` and `lse`, then execution reached later DCP configuration;
- batch 3: `ValueError: too many values to unpack (expected 2)`.

The backend call did not contain `return_lse` on the base.

At exact candidate commit `997071d`, the submitted regression passed (`7 passed`, with four subtests). An independent matrix through the actual forward method showed tensor-only returns for batches 1, 2, and 3 all fail immediately with the candidate's explicit contract error and with `return_lse=True` present in backend kwargs. Valid `(attn_output, lse)` tuples for batches 1 and 3 succeeded, and the exact LSE object reached the mocked `dcp_a2a_lse_reduce` boundary. `(output, None)` and a list container failed before collective use.

Four selected `test_dcp_lse_combine.py` cases ran on the assigned GPU and passed against independent CPU references for natural-log LSE, base-2 LSE, larger batch, and larger head dimension.

## Environment and paths

- Interpreter: `/tmp/amdpilot-repo-j-5d179cc16475/venv/bin/python`
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`
- GPU: one AMD Instinct MI355X, `gfx950:sramecc+:xnack-`
- `sglang` import: `/job/repo/python/sglang/__init__.py`
- reviewed module import: `/job/repo/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py`
- loaded aiter native module: `/tmp/amdpilot-repo-j-5d179cc16475/cache/aiter/module_aiter_core.so`

No native source changed, so a native/FlyDSL rebuild was not applicable.

## Limitations

Real multi-rank collective transport, the issue's 8x MI300X/gfx942 topology, the GLM-5.3 weights, and a full server request were unavailable. The candidate PR body also names mirror issue 597 rather than the reviewed mirror issue 669; that metadata discrepancy does not affect the implementation verdict.
