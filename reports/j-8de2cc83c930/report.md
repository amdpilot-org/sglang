# ROCm router GEMM and expert-bias precision report

## Conclusion

The baseline reproduces both precision defects described in sgl-project/sglang issue 34857:

- `aiter_dsv3_router_gemm` requests the router GEMM output in BF16. A constructed sub-BF16-ULP near-tie selects expert 0 instead of the FP32-reference expert 1.
- The correction bias is cast to the gating dtype before `aiter.biased_grouped_topk`. Nine FP32 bias values separated by `1e-5` collapse to one BF16 value, changing the top-8 set from experts 1-8 to experts 0-7.

The tested candidate, sgl-project/sglang PR 35055 head `c88f4800f47ea048d38c77d22c084f4c45e74ba3`, fixes the output dtype request and bias allocation. On this stock gfx942 stack, the router output dtype becomes FP32, but the untuned Aiter GEMM fallback still returns BF16-rounded values. The bias fix is active and restores the reference top-8 set.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP runtime: `7.2.26015-fc0010cf6a`
- Baseline `main`: `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Tested candidate head: `c88f4800f47ea048d38c77d22c084f4c45e74ba3`
- Delivery commit: `1948e7bb73d62b580d96404f2a7455b2b1662657`

## Code and native paths

- Router helper: `python/sglang/srt/layers/rocm_linear_utils.py`
- Router gate and bias allocation: `python/sglang/srt/models/deepseek_v2.py`
- Expert top-k dispatch: `python/sglang/srt/layers/moe/topk.py`
- Aiter top-k Python source: `/sgl-workspace/aiter/aiter/ops/topk.py`
- Aiter tuned GEMM Python source: `/sgl-workspace/aiter/aiter/tuned_gemm.py`
- Native Aiter core: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Native Aiter custom GEMM: `/sgl-workspace/aiter/aiter/jit/module_custom.so`
- Native Aiter MoE top-k: `/sgl-workspace/aiter/aiter/jit/module_moe_asm.so`

Actual operation names are `aiter_dsv3_router_gemm`, `tgemm.mm`, and `aiter.biased_grouped_topk`. The independent reference uses `torch.matmul` and `torch.topk` in FP32.

## Synthetic method

The harness uses no model weights. It constructs:

- BF16 hidden states of shape `[1, 7168]` and BF16 router weights of shape `[256, 7168]`.
- A router near-tie where experts 0 and 1 differ by `7.450580596923828e-08`, below the BF16 ULP of `6.103515625e-05` at the winning logit.
- A grouped top-k case with 256 experts, 8 groups, `topk_group=4`, and `topk=8`. Nine FP32 biases are `0.1 + i * 1e-5`; all nine round to BF16 `0.10009765625`.

The FP32 router reference is `torch.matmul(hidden_states.float(), weight.t().float())`. The FP32 bias reference computes sigmoid scores, adds the FP32 bias, applies the grouped-top-k mask, and selects with `torch.topk`.

## Raw results

### Router GEMM

| Run | Helper output dtype | Max abs error | Max rel error | Mean abs error | Reference argmax | Actual argmax |
|---|---:|---:|---:|---:|---:|---:|
| Baseline `main` | BF16 | 9.909272193908691e-06 | 7.307876949198544e-04 | 7.712515071034431e-08 | 1 | 0 |
| Candidate PR 35055 | FP32 | 9.909272193908691e-06 | 7.307876949198544e-04 | 7.712515071034431e-08 | 1 | 0 |
| Delivery branch | FP32 | 9.909272193908691e-06 | 7.307876949198544e-04 | 7.712515071034431e-08 | 1 | 0 |

The direct `tgemm.mm(..., otype=torch.float32)` output is bitwise equal to the direct BF16 output on this stack. Aiter logs:

```text
M:1, N:256, K:7168, dtype=torch.bfloat16, otype=torch.float32:
not found tuned config ... using skinny solution:2
```

The `skinny_gemm` implementation allocates its output with `dtype=inp.dtype` and only converts to the requested `otype` afterward, so the FP32 request cannot recover information already rounded by the fallback kernel. This explains why the candidate changes the output dtype but not the near-tie result on stock gfx942.

The top-8 set is unchanged in this router case because both near-tie experts remain selected; the argmax ordering is the sensitive result.

### Expert correction bias

| Run | Gating dtype | Bias dtype at add | Reference top-8 | Actual top-8 | Set matches |
|---|---|---|---|---|---:|
| Current BF16 behavior | BF16 | BF16 | 1,2,3,4,5,6,7,8 | 0,1,2,3,4,5,6,7 | No |
| Candidate behavior | FP32 | FP32 | 1,2,3,4,5,6,7,8 | 1,2,3,4,5,6,7,8 | Yes |

The baseline source casts `correction_bias.to(dtype=gating_output.dtype)` before the Aiter top-k call. With the candidate, the bias is allocated FP32 and the router output is FP32, so the cast is a no-op. The current grouped Aiter path also upcasts FP32 gating when the bias is FP32; the baseline defeats that protection by allocating the bias as BF16 for Aiter plus FP8/compressed-tensors/quark quantization.

## Candidate and delivery

The tested candidate is the open upstream PR 35055, head commit `c88f4800f47ea048d38c77d22c084f4c45e74ba3` (original fix commit `03093d7c196dc83ff31d5ac1a6dd9ae656dcfdfb`). The delivery branch cherry-picks that fix onto the mirror's current `main` and preserves the newer `expert_pack` zero-initialization behavior.

The candidate's CPU dtype test passes:

```text
2 passed, 3 warnings in 13.74s
```

No existing numerical gate was changed. The only added test pins the correction-bias allocation dtype.

## Commands

```bash
git clone --depth 50 https://github.com/amdpilot-org/sglang.git /job/sglang
git fetch https://github.com/sgl-project/sglang.git pull/35055/head:candidate-35055
git checkout -b amdpilot/j-8de2cc83c930 origin/main
git cherry-pick 03093d7c196dc83ff31d5ac1a6dd9ae656dcfdfb
AITER_LOG_TUNED_CONFIG=1 /opt/venv/bin/python reports/j-8de2cc83c930/validate_rocm_router_precision.py
/opt/venv/bin/python -m pytest -q test/registered/unit/models/test_deepseek_v2_moe_gate_correction_bias_dtype.py
rocm-smi --showproductname --csv
rocminfo
```

## Not done

- No full model weights were downloaded.
- No end-to-end GLM-5.2 accuracy evaluation was run.
- No Aiter GEMM tuning job was run; the stock gfx942 fallback limitation is reported rather than hidden.
- No existing numerical gate was modified.

Raw JSON and logs are retained alongside this report in `reports/j-8de2cc83c930/`.
