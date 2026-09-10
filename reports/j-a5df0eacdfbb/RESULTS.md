# Bounded gfx942 BF16/FP8 KV reduced-attention study

## Scope

This study exercises the existing AITER `unified_attention` reduced-attention path on one assigned MI300X (`gfx942`). It compares BF16 and FP8 KV outputs against independent dequantized references, records actual cache/kernel allocation and dispatch, and reports shared-hardware timing uncertainty. It makes no model-quality claim.

## Environment

- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- GPU: `AMD Instinct MI300X`, capability `(9, 4)`, 304 CUs, 206141652992 bytes
- SGLang source: `/job/sglang`
- AITER source: `/sgl-workspace/aiter/aiter/ops/triton/attention/unified_attention.py`
- AITER native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`

## Installed-source baseline

Before cloning or editing, the preinstalled source was exercised with `mla_kv_pack_quantize_fp8` on one MI300X. The recorded baseline is saved at `/job/baseline-first.json` and is explicitly labeled as installed-source evidence, not proof for later checkout changes.

- Dimensions: batch `1024`, heads `32`, QK nope `128`, QK rope `64`, V head `128`
- Input dtype: BF16; output dtype: FP8 e4m3
- Scales: `k_scale_inv=0.7`, `v_scale_inv=1.3`
- Reference: independent `torch.cat`/expand + float32 multiply + FP8 cast
- Accuracy: K and V byte-equal to reference, max absolute FP8 code difference `0.0`
- Dispatch: `v0`, `BLOCK_S=16`, `num_warps=4`, `num_stages=3`
- Timing: CUDA events, 5 warmups + 30 measured calls
- Median: `0.092012 ms`; mean `0.095985 ms`; stdev `0.013182 ms`
- First-GPU-execution elapsed: `1.825809 s`

## Method

Four existing sequence-length configurations were selected for one measured block bottleneck: page size `16`, head dimension `256`, batch `4`, query heads `16`, and KV heads `1`. Sequence lengths were `1024`, `2048`, `4096`, and `8192`. Each case used identical generated BF16 base tensors for BF16 and FP8; FP8 tensors were quantized from the same base with explicit per-tensor scales.

BF16 outputs were compared to an independent float32 einsum/softmax reference. FP8 outputs were compared to an independent reference that dequantized q/k/v with the explicit scales before the same float32 einsum/softmax. Accuracy gates were fixed before measurement:

- BF16: finite, `max_abs <= 0.02`, `cosine >= 0.9999`
- FP8: finite, `mismatch_fraction <= 0.005`, `cosine >= 0.99`

Timing used CUDA events around real forward calls: 5 warmups and 20 measured calls per run. All cases ran on the same MI300X, so timing uncertainty is reported as sample standard deviation and a 95% normal confidence interval for the mean.

## Results

| Case | BF16 max abs | BF16 cosine | BF16 median ms | BF16 mean ms | BF16 stdev ms | BF16 95% CI ms | FP8 max abs | FP8 cosine | FP8 median ms | FP8 mean ms | FP8 stdev ms | FP8 95% CI ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.000000 | 1.000000 | 0.101815 | 0.103731 | 0.008600 | 0.003769 | 0.002604 | 0.999999 | 0.106005 | 0.109725 | 0.013704 | 0.006006 |
| 2048 | 0.000000 | 1.000000 | 0.108289 | 0.109537 | 0.004263 | 0.001869 | 0.000000 | 1.000000 | 0.110695 | 0.113037 | 0.008243 | 0.003613 |
| 4096 | 0.000000 | 1.000000 | 0.107809 | 0.109651 | 0.007880 | 0.003454 | 0.002604 | 0.999999 | 0.113241 | 0.114210 | 0.004071 | 0.001784 |
| 8192 | 0.000000 | 1.000000 | 0.107508 | 0.109882 | 0.008273 | 0.003626 | 0.003125 | 0.999999 | 0.113141 | 0.114739 | 0.006179 | 0.002708 |

All BF16 and FP8 runs passed their fixed numerical gates. No case was rejected.

## Dispatch and allocation

All four cases dispatched `kernel_unified_attention_3d`. BF16 used `TILE_SIZE=16`; FP8 used `TILE_SIZE=32`. Sequence lengths 1024 used `NUM_SEGMENTS=64`; 2048, 4096, and 8192 used `NUM_SEGMENTS=128`. All runs used `num_warps=2` and `num_stages=2`.

Cache and kernel workspace allocations stayed well below the 48 GiB live-allocation limit. The largest observed peak allocation was `230910464` bytes for the FP8 8192 case. No model weights were downloaded or generated.

## Reproduction

From `/job/sglang`, run:

```bash
/opt/venv/bin/python reports/j-a5df0eacdfbb/bench_fp8_kv_attention.py
```

The script writes `reports/j-a5df0eacdfbb/results.json`. The installed-source baseline is recorded in `/job/baseline-first.json`.

## Limits

- Workload cases: 4
- Kernel/dispatch configurations: 4
- Warmup calls per run: 5
- Measured calls per run: 20
- Generated weights: 0 bytes
- Total live allocation limit: 48 GiB
- Wall limit: 7200 seconds

## Notes

- The first benchmark draft accidentally evaluated an uninitialized output tensor before the first kernel call, producing non-finite accuracy values. This was corrected by executing one real forward before accuracy comparison; the final results above are from the corrected run.
- The installed-source baseline and the checkout benchmark are separate evidence. The baseline is not proof for later checkout changes.
