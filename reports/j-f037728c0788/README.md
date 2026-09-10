# gfx942 DSpark chunk-partition study

## Scope

This is a bounded follow-up to the reduced DSpark pipeline study in mirror PR 463. It does not repeat that completed unchunked pipeline case or the decode-sized block study in mirror PR 469. Instead, it uses one identical synthetic workload and six legal target-hidden/commit-hidden row partitions to compare chunked execution with the unchunked production-primitive result.

The study uses one assigned AMD Instinct MI300X (`gfx942:sramecc+:xnack-`), the qualified Torch/ROCm stack, and locally generated weights. It makes no GLM-weight, full-model, distributed, graph-replay, scheduler-overlap, or end-to-end serving claim.

## Installed-source baseline

The first GPU execution used the preinstalled interpreter and the existing speculative kernel test:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/python/sglang/kernels/aot/tests/speculative/test_eagle_utils.py
```

The test passed. Its `verify_tree_greedy` result matched the independent expected outputs exactly. A separate bounded timing probe made 20 synchronized calls and recorded a median of `0.019288 ms`. First GPU execution, including tensor setup and synchronization, took `0.268875 s`. The full artifact is `/job/baseline-first.json`.

This baseline is labeled installed-source evidence only. It is not proof for the persistent checkout changes used below.

## Persistent checkout

- Base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Branch: `amdpilot/j-f037728c0788`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- SGLang source: `/job/sglang/python/sglang/__init__.py`
- Native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- Image: operator-provided local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`. The `docker` CLI is unavailable inside the job, so this ID is recorded verbatim rather than re-inspected.

Focused primitive validation passed:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_dspark_kernel_parity.py \
  test/registered/spec/utils/test_build_eagle_tree.py
```

Result: `3 passed`, `22 subtests passed`.

## Workload

All six cases use the same seeded synthetic data:

- `32768` target hidden rows
- `128` spatial batch
- `gamma=5`, `verify_num_draft_tokens=6`
- target hidden size `4096`
- draft hidden size `1024`
- vocabulary size `8192`
- three commit stages with head dimension `576`
- bfloat16 hidden states and weights
- seed `30734`

The partition matrix is:

| Case | Target chunks | Commit chunks |
|---|---:|---:|
| `full_full` | 1 | 1 |
| `half_full` | 2 | 1 |
| `full_half` | 1 | 2 |
| `half_half` | 2 | 2 |
| `quarter_quarter` | 4 | 4 |
| `eighth_eighth` | 8 | 8 |

Each case performs one cold forward and five warm forwards. Cold execution captures first-use compilation/launch effects; warm throughput uses the median of five `time.perf_counter` measurements, each ending in `torch.cuda.synchronize()`. Raw CUDA-event timings are also retained in `results.json`.

## Numerical gates

The gates are unchanged from the prior reduced-pipeline study:

- target projection maximum absolute error versus an independent FP32 matmul: `<= 0.02`
- commit projection maximum absolute error versus an independent FP32 matmul: `<= 0.02`
- draft hidden selection: exact equality with an independent view
- greedy acceptance: exact equality with an independent leading-match/cumprod reference
- accept finalization: exact equality with independent integer arithmetic
- output-token commit: exact equality with an independent scatter reference
- chunked final state and output tokens: exact equality with the unchunked result
- total useful work: exact equality with the unchunked result

## Results

All six cases passed every numerical and resource gate. Each case produced the same final state and `33052` useful tokens (`32768` projected target tokens plus `284` accepted tokens).

| Case | Warm median wall (s) | Useful tokens/s | Target vs unchunked | Commit vs unchunked | Final state |
|---|---:|---:|---:|---:|---|
| `full_full` | `0.00381797` | `8656962.93` | `0` | `0` | exact |
| `half_full` | `0.00415604` | `7952764.46` | `0` | `0` | exact |
| `full_half` | `0.00383875` | `8610085.92` | `0` | `0.001953125` | exact |
| `half_half` | `0.00407816` | `8104629.79` | `0` | `0.001953125` | exact |
| `quarter_quarter` | `0.00410912` | `8043565.00` | `0` | `0.001953125` | exact |
| `eighth_eighth` | `0.00429294` | `7699161.29` | `0` | `0.001953125` | exact |

The first case's cold wall time was `1.62149 s`, reflecting first-use compilation/initialization. Later cold wall times were between about `0.00363 s` and `0.00430 s`. The complete measured study took `5.68335 s` after import and used `36` real forwards total.

Generated weights were `70647808` bytes. Absolute peak allocated memory was `5333453824` bytes, below the `48` GB limit. The per-case peak-above-start values ranged from `1086547456` to `3761638400` bytes; these values include the retained unchunked comparison result and are not isolated per-case allocations.

## Reproduction

Use job-private caches outside the repository:

```bash
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-f037728c0788/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-f037728c0788/inductor
export HF_HOME=/tmp/sglang-cache-j-f037728c0788/hf
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$HF_HOME"
```

Run the study:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-f037728c0788/chunk_partition.py \
  --output reports/j-f037728c0788/results.json
```

Raw values are in `results.json`.

## Boundary

The model-specific GLM/DSpark target-hidden projector constructor was not instantiated because it requires a full model configuration and checkpoint. No checkpoint was downloaded. The reduced path covers target hidden projection, draft hidden selection, commit-KV projection, greedy acceptance, finalization, and output-token commit only.

Upstream issue 30734, its comments, and its currently linked PRs 31047, 31260, 31451, 31457, 32186, and 32374 were read as context only. No upstream issue, pull request, or comment was posted or changed.
