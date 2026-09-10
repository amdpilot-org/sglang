# MI300X BF16/FP8 KV decode evidence

## Result

The supported Triton MHA decode path was exercised on one AMD Instinct MI300X
(`gfx942`) with locally generated BF16 model weights. Six workload cases were
run: three finite batch/context shapes for BF16 KV and the same three for FP8
KV. Every recurrent decode step passed an independent cache-row reference
check. FP8 used explicit scales `k_scale=0.03125` and `v_scale=0.0625`; BF16
used unit scales. No model-quality claim is made.

The DSA FP8 MLA path was not used as the primary case because its HIP storage
branch writes raw FP8 without the per-block scales used by the non-HIP DSA
layout. The Triton MHA path is the meaningful supported neighboring control
because it accepts explicit K/V scales and passes their descales to the decode
kernel.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Declared local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440003949`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Triton: `3.7.0`
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Checkout SGLang path: `/job/sglang/python/sglang/__init__.py`
- Native/tool paths: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`,
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/flash_mla.py`,
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`, and
  `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`

## Installed-source baseline

Before cloning or editing, the installed source at commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` ran:

```bash
cd /sgl-workspace/sglang/test/registered/attention
/opt/venv/bin/python -m unittest \
  test_wave_attention_kernels.TestWaveAttention.test_grouped_decode_attention
```

The test passed all 24 wave-versus-Triton grouped-decode comparisons. Its gates
were cosine similarity greater than `0.99` and `torch.allclose(atol=3e-2)`.
Observed cosine similarities ranged from `0.9970703125` to `1.0009765625`.
The first GPU execution took 60 seconds by UTC epoch timestamps; unittest
reported 53.390 seconds. The first attempted command used `/usr/bin/time`,
which is absent in the image and exited 127 before GPU work. The complete
record is in `/job/baseline-first.json`. This baseline describes the installed
source only and is not evidence for later checkout changes.

## Method

The benchmark uses the checkout's dense-attention fixture with:

- 4 query heads, 4 KV heads, head dimension 64, hidden size 256
- page size 16 and shuffled physical pages
- BF16 model dtype for both cache dtypes
- three recurrent decode steps per workload
- two warmup forwards and ten timed forwards per workload
- explicit FP8 K/V scales `0.03125` and `0.0625`

The independent reference reads the actual K/V cache rows selected by the page
table, dequantizes FP8 rows with the explicit scales, computes attention in
PyTorch, and applies the same output projection. The gate is
`torch.allclose(atol=3e-2, rtol=3e-2)` plus finite-output checks. Timing uses
`time.perf_counter_ns` around the full eager forward and a CUDA synchronize.
Kernel dispatch is recorded by wrapping `decode_attention_fwd` and by one ROCm
profiler pass per workload.

## Numerical and timing observations

| Shape | KV dtype | Max error vs cache reference | Median ms | P90 ms | P99 ms |
|---|---|---:|---:|---:|---:|
| batch 1, context 128–130 | BF16 | 0.00048828125 | 0.440343 | 0.528346 | 0.606122 |
| batch 1, context 128–130 | FP8 | 0.0 | 0.516072 | 0.568749 | 0.691596 |
| batch 4, contexts 128–132 | BF16 | 0.00048828125 | 0.439135 | 0.588740 | 0.595527 |
| batch 4, contexts 128–132 | FP8 | 0.000244140625 | 0.528529 | 0.705613 | 0.764621 |
| batch 8, contexts 64–73 | BF16 | 0.00048828125 | 0.429349 | 0.455672 | 0.507350 |
| batch 8, contexts 64–73 | FP8 | 0.0009765625 | 0.488459 | 0.544170 | 0.683943 |

The largest direct BF16-versus-FP8 output difference was `0.005017757415771484`
on the batch-8 shape. This is an observation from these synthetic cases, not a
quality or general-performance conclusion.

## Allocation observations

| Shape | KV dtype | Predicted KV bytes | Predicted fixture live bytes | Final live bytes | Peak live bytes |
|---|---|---:|---:|---:|---:|
| batch 1 | BF16 | 212,992 | 2,033,868 | 82,322,944 | 83,619,840 |
| batch 1 | FP8 | 106,496 | 1,927,372 | 82,433,024 | 83,729,920 |
| batch 4 | BF16 | 704,512 | 2,557,140 | 83,260,416 | 84,586,496 |
| batch 4 | FP8 | 352,256 | 2,204,884 | 83,618,304 | 84,944,384 |
| batch 8 | BF16 | 835,584 | 2,724,260 | 84,146,176 | 85,507,584 |
| batch 8 | FP8 | 417,792 | 2,306,468 | 84,569,600 | 85,931,008 |

The cache pool stores 1,024 bytes per slot for BF16 and 512 bytes per slot for
FP8, so FP8 cache allocation is exactly half of BF16 for every shape. The
largest observed peak live allocation was 85,931,008 bytes, far below the 48 GiB
guard. Absolute live bytes include process-global allocations. The JSON also
records baseline-relative deltas; those remain larger than the fixture-only
prediction because Triton JIT and allocator state are process-global.

## Dispatch evidence

Every recurrent step called `TritonAttnBackend.forward_decode` exactly once and
dispatched `decode_attention_fwd` with:

- BF16 cache dtype and descales `1.0`/`1.0`
- FP8 cache dtype and descales `0.03125`/`0.0625`
- `enable_lean=false`

The timing phase recorded 12 dispatches (ten timed forwards plus two output
consistency forwards), and each profiler pass recorded exactly one dispatch.

The ROCm profiler observed the expected Triton attention kernels
`_fwd_kernel_stage1` and `_fwd_kernel_stage2`, plus
`create_flashinfer_kv_indices_triton`, `get_num_kv_splits_triton`, and
`store_kvcache`. FP8 profiles also contain BF16-to-FP8 copy and scale-multiply
kernels on the write path. Full event names, counts, and self device times are
in `results.json`.

## Upstream context

Read-only inspection of sgl-project/sglang issue 31783 found:

- PR 18882, FP8 KV support for the Triton attention backend, is merged.
- PR 31652, fusing FP8 KV quantization into the store kernel, is open.
- PR 31660, FP8 KV scale warning handling, is closed.
- Issue 30815 tracks unfused FP8 KV decode overhead.

This change does not duplicate those fixes: it adds a bounded evidence script
and raw results for the already-supported Triton path. No upstream issue, PR,
or comment was posted or modified.

## Reproduction

```bash
cd /job/sglang
PYTHONPATH=$PWD/python /opt/venv/bin/python \
  reports/j-198dad0d6344/benchmark_fp8_kv.py \
  --output reports/j-198dad0d6344/results.json
```

Raw output is in `results.json` and the captured run log is in `run.log`.
Two earlier partial runs reached GPU work but failed in report-only allocation
traversal code (`root`/`roots`, dictionary mutation, and an overly broad walk
into Transformers). Those partial runs are not used as results; the final
unchanged workload matrix completed successfully.
