# MI300X FP8 blockwise dense-linear study

## Scope

- Campaign: `repo-e2e-20260909`
- Upstream context: `sgl-project/sglang` issue `15194`
- Coordination tracker: `amdpilot-org/amdpilotv2` issue `402`
- Hardware: one assigned AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Branch: `amdpilot/j-034210b05818`
- PR base: `main`

This is a bounded, measurement-only study of existing supported dense-linear kernels. It does not modify kernel code or dispatch logic.

## Environment

- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- SGLang source: `/job/sglang/python/sglang`
- Aiter native module: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`
- Torch native package: `/opt/venv/lib/python3.10/site-packages/torch`
- GPU UUID: `35383633-3433-6633-3832-613639663164`

## Installed-source baseline

The first relevant installed-source attempt was the upstream W8A8 INT8 dense-linear test. It failed before kernel dispatch with:

```text
NameError: name 'int8_scaled_mm' is not defined
```

The supported neighboring control was the upstream Block-INT8 dense-linear test. It passed all three subtests against an independent dequantized reference.

- Command:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/unit/layers/quantization/test_int8_linear_methods.py::TestBlockInt8Linear::test_block \
  -s -p no:cacheprovider
```

- Result: `1 passed, 3 subtests passed`
- First relevant attempt elapsed: `19.468 s`
- Supported control elapsed: `22.475 s`
- Accuracy gate: `rtol=5e-2`, `atol=1e-1`

The full installed-source baseline record is saved at `/job/baseline-first.json`.

## Study design

- Shape: `N=4096`, `K=512`
- Workload M values: `1`, `64`, `512`, `2048`
- Input dtype: `torch.bfloat16`
- Weight dtype: `torch.float8_e4m3fn`
- Weight-scale dtype: `torch.float32`
- Output dtype: `torch.bfloat16`
- Weight-block size: `[128, 128]`
- Accuracy gate: `rtol=5e-2`, `atol=1e-1`
- Timing method: 10 warmup forwards, then 30 timed real forwards per case, using one CUDA event pair around each forward
- Total workload cases: 4
- Total configurations: 4
- Weight size: about `2 MB`
- Maximum observed live allocation: `381682176 bytes`
- Study wall time: `4.409790992736816 s`

### Configurations

1. `aiter_blockwise_fp8`
2. `triton_tuned_m1`
3. `triton_tuned_m512`
4. `triton_default`

All four configurations use the same synthetic packed weight and scale, and each M case reuses the same input and independent dequantized reference across all configurations.

## Results

All numerical gates passed. No configuration was rejected for numerical regression.

| M | Configuration | Max abs error | Median ms | CV | Approx 95% CI ms |
|---:|---|---:|---:|---:|---:|
| 1 | `aiter_blockwise_fp8` | 0.02333252876996994 | 0.8283119797706604 | 0.005349802711862937 | 0.001587075565107812 |
| 1 | `triton_tuned_m1` | 0.020320385694503784 | 0.1347310021519661 | 0.08791666117547696 | 0.00434277241086189 |
| 1 | `triton_tuned_m512` | 0.020320385694503784 | 0.20064300298690796 | 0.01760503737907525 | 0.001268439822278551 |
| 1 | `triton_default` | 0.020320385694503784 | 0.397938996553421 | 0.009421907899533158 | 0.0013447528673415198 |
| 64 | `aiter_blockwise_fp8` | 0.026611804962158203 | 0.8556750118732452 | 0.012727238940330772 | 0.003911187612453205 |
| 64 | `triton_tuned_m1` | 0.028577178716659546 | 0.13563349843025208 | 0.027764125483203905 | 0.0013550396297971772 |
| 64 | `triton_tuned_m512` | 0.028577178716659546 | 0.2088019996881485 | 0.016820836766254486 | 0.0012615229468086666 |
| 64 | `triton_default` | 0.028577178716659546 | 0.41086798906326294 | 0.009713015000742162 | 0.0014306193525917266 |
| 512 | `aiter_blockwise_fp8` | 0.029354214668273926 | 1.158614456653595 | 0.013228776659662916 | 0.0054848340537816155 |
| 512 | `triton_tuned_m1` | 0.031604886054992676 | 0.41858650743961334 | 0.0069767543344419715 | 0.001046860533440336 |
| 512 | `triton_tuned_m512` | 0.031604886054992676 | 0.33848150074481964 | 0.013355982580198407 | 0.0016210064737820678 |
| 512 | `triton_default` | 0.031604886054992676 | 0.5550810098648071 | 0.01123334286584046 | 0.00224009887105842 |
| 2048 | `aiter_blockwise_fp8` | 0.0335288941860199 | 3.1224470138549805 | 0.012692480376862016 | 0.014223885845128072 |
| 2048 | `triton_tuned_m1` | 0.03112843818962574 | 1.359919011592865 | 0.00167479277792805 | 0.000815374816449089 |
| 2048 | `triton_tuned_m512` | 0.03112843818962574 | 0.9379650056362152 | 0.009347778383682637 | 0.0031424571951397715 |
| 2048 | `triton_default` | 0.03112843818962574 | 1.8711789846420288 | 0.011670476235685467 | 0.007816693539836708 |

For this shape, `triton_tuned_m1` is fastest at `M=1` and `M=64`; `triton_tuned_m512` is fastest at `M=512` and `M=2048`. Aiter is numerically valid but slower than the Triton paths for this specific `N=4096,K=512` block.

## Reproduction

```bash
cd /job/sglang
SGLANG_USE_AITER=1 /opt/venv/bin/python \
  reports/j-034210b05818/run_fp8_blockwise_study.py \
  reports/j-034210b05818/results.json
```

Raw results are in `reports/j-034210b05818/results.json`.

## Limits and uncertainty

- One GPU was used for all measurements.
- Timing uncertainty is reported as coefficient of variation and an approximate 95% confidence interval from 30 samples.
- The study is bounded to four configurations and four M cases.
- No artificial burn, unbounded loop, or repeated work solely to occupy the GPU was used.
- No upstream issue, PR, or comment was posted or modified.
