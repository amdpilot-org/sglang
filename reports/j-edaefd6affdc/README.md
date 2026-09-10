# gfx942 synthetic diffusion feature-block study

## Scope

This is a reduced, prefill-sized throughput study of a synthetic Wan-style
feature block on one AMD Instinct MI300X (`gfx942`). It is not a
generated-video quality test, and it is distinct from earlier standalone
operator boundary probes.

The block is:

```text
causal Conv3d(3x3x3) -> RMSNorm + SiLU -> causal Conv3d(3x3x3)
```

The study compares:

- an independent Torch formula on contiguous `NCDHW` tensors,
- the same independent formula on `channels_last_3d` tensors,
- the existing SGLang `WanCausalConv3d` and `wan_rmsnorm_silu` operators on
  `channels_last_3d` tensors.

## Environment

- GPU: AMD Instinct MI300X (`gfx942`)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Python: `/opt/venv/bin/python`
- Main-branch commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Upstream candidate commit: `0e2a6f4dffd4d00575b571da07385bd9162f6d1a`
  (sgl-project/sglang PR 38650)

## Cases

All cases use `bfloat16`, four frames, and locally generated weights. The six
cases are:

| Case | Shape | Tokens |
|---|---|---:|
| `c96_128` | `[1, 96, 4, 128, 128]` | 65,536 |
| `c96_256` | `[1, 96, 4, 256, 256]` | 262,144 |
| `c96_384` | `[1, 96, 4, 384, 384]` | 589,824 |
| `c96_512` | `[1, 96, 4, 512, 512]` | 1,048,576 |
| `c96_512_b2` | `[2, 96, 4, 512, 512]` | 2,097,152 |
| `c384_256` | `[1, 384, 4, 256, 256]` | 262,144 |

## Accuracy gate

The unchanged gate is:

```text
max_abs_error <= 0.15
rmse <= 0.03
```

All six main-branch cases and all six upstream-candidate cases pass.

## Timing method

Each path uses:

- three warmup forwards,
- ten timed forwards,
- CUDA events for warm timing,
- wall-clock time for the first cold forward.

The independent reference is:

```python
F.conv3d(
    F.pad(x, (1, 1, 1, 1, 2, 0)),
    conv1.weight,
    conv1.bias,
)
```

followed by `F.silu(F.normalize(...) * scale * gamma)` and the second
convolution.

## Main-branch results

| Case | Contiguous warm (s) | Channels-last warm (s) | Fused warm (s) | Fused cold (s) | Max error | RMSE | Peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c96_128` | 0.000470 | 0.000693 | 0.000907 | 0.731 | 0.015625 | 0.002862 | 0.372 |
| `c96_256` | 0.001409 | 0.002361 | 0.003525 | 0.00356 | 0.023438 | 0.002877 | 1.481 |
| `c96_384` | 0.003381 | 0.005399 | 0.008732 | 0.00884 | 0.017578 | 0.002895 | 3.327 |
| `c96_512` | 0.005748 | 0.009775 | 0.015041 | 0.1809 | 0.031250 | 0.002886 | 5.911 |
| `c96_512_b2` | 0.011919 | 0.019699 | 0.030903 | 0.0300 | 0.023438 | 0.002883 | 6.757 |
| `c384_256` | 0.012196 | 0.016959 | 0.022490 | 0.0656 | 0.046875 | 0.005710 | 5.955 |

## Layout-conversion cost

The benchmark separately times:

- activation `NCDHW -> channels_last_3d`,
- activation `channels_last_3d -> NCDHW`,
- post-convolution output `NCDHW -> channels_last_3d`,
- convolution weight `NCDHW -> channels_last_3d`.

Representative main-branch costs are:

| Case | Activation to channels-last (s) | Activation to contiguous (s) | Post-conv to channels-last (s) | Weight to channels-last (s) |
|---|---:|---:|---:|---:|
| `c96_128` | 0.0000314 | 0.0000320 | 0.0000319 | 0.0000111 |
| `c96_256` | 0.000117 | 0.000133 | 0.000116 | 0.0000138 |
| `c96_384` | 0.000825 | 0.001120 | 0.000825 | 0.0000143 |
| `c96_512` | 0.000958 | 0.002437 | 0.000957 | 0.0000111 |
| `c96_512_b2` | 0.001912 | 0.004706 | 0.001909 | 0.0000223 |
| `c384_256` | 0.001247 | 0.000642 | 0.001245 | 0.0000101 |

## Upstream candidate

The closest upstream candidate is sglang PR 38650 at commit
`0e2a6f4dffd4d00575b571da07385bd9162f6d1a`. It fuses the convolution bias
epilogue and tiles the RMSNorm+SiLU kernel.

The same six cases pass the unchanged accuracy gate. Warm fused-path times are
similar to main, with modest improvements on the larger `C=96` cases:

| Case | Main fused warm (s) | Candidate fused warm (s) |
|---|---:|---:|
| `c96_128` | 0.000907 | 0.000903 |
| `c96_256` | 0.003525 | 0.003388 |
| `c96_384` | 0.008732 | 0.008402 |
| `c96_512` | 0.015041 | 0.014413 |
| `c96_512_b2` | 0.030903 | 0.028638 |
| `c384_256` | 0.022490 | 0.022508 |

The candidate does not change the overall conclusion for this synthetic block:
the fused `channels_last_3d` path remains slower than the independent
contiguous Torch formula on `gfx942`.

## Commands

Run the main-branch study:

```bash
export PYTHONPATH=/job/sglang/python
/opt/venv/bin/python reports/j-edaefd6affdc/benchmark.py \
  --output reports/j-edaefd6affdc/results-main.json
```

Run the upstream-candidate study:

```bash
git worktree add --detach /tmp/sglang-pr38650 \
  0e2a6f4dffd4d00575b571da07385bd9162f6d1a
cp reports/j-edaefd6affdc/benchmark.py \
  /tmp/sglang-pr38650/reports/j-edaefd6affdc/benchmark.py
cd /tmp/sglang-pr38650
export PYTHONPATH=/tmp/sglang-pr38650/python
/opt/venv/bin/python reports/j-edaefd6affdc/benchmark.py \
  --output /tmp/gfx942-feature-block-pr38650.json
```

## Limits

- Maximum cases: 6
- Maximum locally generated weight storage: 4 GiB
- Maximum live allocations: 48 GiB
- Wall-clock limit: 7,200 seconds
- No full model weights are downloaded.
- No upstream issue, PR, or comment is modified.

## Conclusion

On one MI300X, the existing supported diffusion-side convolution and
normalization operators are numerically accurate for this reduced block, but
the `channels_last_3d` fused path is not faster than the independent
contiguous Torch formula. Explicit layout conversion is measurable and is
included in the raw results, but the dominant cost is the convolution path
itself rather than the conversion alone.
