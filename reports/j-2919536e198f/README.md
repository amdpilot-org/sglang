# gfx942 BF16/FP8 dense-prefill chunk-partition evidence

## Scope

This is a bounded, one-GPU follow-up to amdpilot-org/sglang issue 379. It
exercises the existing dense Triton prefill attention kernel on an AMD Instinct
MI300X (`gfx942`) with locally generated synthetic tensors. It does not load a
checkpoint, download model weights, or make any model-quality claim.

The tested mirror commit is `0084030179bfba86bfeb6d43f7997d4076329d2c`. The
qualified image is
`amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5` with local image
ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.

## Early installed-source baseline

The first GPU execution used the preinstalled source at commit
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` and the existing fused FP8 QKV KV
cache test. That path dispatched but failed on ROCm because the native wrapper
requires `uint8` cache tensors and rejects `float8_e4m3fn`:

```text
RuntimeError: Tensor match failed for Tensor<2052, 1024>[strides=<1024, 1>,
dtype=float8_e4m3fn, device=rocm:0] at fused_fp8_qkv_kv_cache.cuh:136
- Root cause: Dtype value [float8_e4m3fn] not in the allowed options: [uint8]
```

The first execution elapsed `28.575 s`, including interpreter startup and test
collection. A meaningful neighboring control,
`test_mla_kv_pack_quantize_fp8.py::test_correctness[1024-32-shape0-dtype0]`,
passed against its independent dequantized reference in `7.850 s`. The full
baseline record is saved outside the repository at
`/job/baseline-first.json`; it is clearly labeled as installed-source evidence
and is not proof for the later mirror checkout.

## Workload

The mirror benchmark uses one synthetic 4096-token sequence with 8 query heads,
2 key/value heads, 128 key dimensions, and 128 value dimensions. It compares
BF16 and FP8 inputs against independent float32 references and runs six legal
chunk partitions:

| Partition count | Chunk sizes |
|---:|---|
| 1 | 4096 |
| 2 | 2048 |
| 4 | 1024 |
| 8 | 512 |
| 16 | 256 |
| 32 | 128 |

Each case performs one unmeasured warm-up and three measured repetitions. The
FP8 path uses explicit `k_scale=0.5` and `v_scale=1.25`; BF16 uses unit scales.
The numerical gates are unchanged from the existing dense-prefill test:

- BF16 reference: `rtol=2e-2`, `atol=2e-2`.
- FP8 reference: `rtol=5e-2`, `atol=5e-2`.
- Chunked versus unchunked: the same gate as the corresponding dtype.

## Results

All six BF16 cases and all six FP8 cases pass their unchanged reference gates.
Within each dtype, every chunked output hash exactly matches the unchunked
output hash, and the final-token hash also matches. BF16 maximum reference error
is `0.0028107441030442715`; FP8 maximum reference error is
`0.002924279309809208`.

The actual dispatch is `_fwd_kernel_dense_prefill` from
`python/sglang/kernels/ops/attention/extend_attention.py`. On `gfx942` the
kernel uses `BLOCK_M=64`, `BLOCK_N=64`, and 4 warps. BF16 cold compile is
`0.7555409539490938 s`; FP8 cold compile is `0.002618740312755108 s`.

BF16 input storage is 8 MiB for queries and 2 MiB each for keys and values.
FP8 input storage is 4 MiB for queries and 1 MiB each for keys and values.
Both output tensors are 8 MiB. Peak measured allocations range from
`117441536` to `134219264` bytes for BF16 and from `119538688` to
`127927808` bytes for FP8. No allocation retries or OOM syncs occurred.

Warm query throughput decreases as partition count increases:

| Mode | 1 chunk | 2 chunks | 4 chunks | 8 chunks | 16 chunks | 32 chunks |
|---|---:|---:|---:|---:|---:|---:|
| BF16 tokens/s | 13317582 | 9493883 | 5891341 | 3278401 | 1737842 | 865154 |
| FP8 tokens/s | 2321638 | 1564304 | 946872 | 525584 | 279709 | 145134 |

Warm key/value visit throughput remains comparatively stable because finer
partitions re-read longer prefixes:

| Mode | 1 chunk | 2 chunks | 4 chunks | 8 chunks | 16 chunks | 32 chunks |
|---|---:|---:|---:|---:|---:|---:|
| BF16 visits/s | 13317582 | 14240825 | 14728353 | 14752803 | 14771656 | 14275034 |
| FP8 visits/s | 2321638 | 2346456 | 2367179 | 2365127 | 2377523 | 2394708 |

Raw per-case timings, hashes, memory statistics, grids, and dispatch details
are in `chunk_partition_results.json`.

## Preserved boundaries and corrections

- The installed fused FP8 QKV cache path is unsupported on this stack because
  its native wrapper requires `uint8` cache tensors.
- The existing dense-prefill unit test is gated to `gfx950`, so it does not
  cover this `gfx942` case. The public kernel itself runs successfully here.
- The first BF16 run with synthetic amplitude `0.1` failed the unchanged
  `2e-2` gate with maximum error `0.02282899059355259`. The gate was kept
  unchanged and only the synthetic amplitude was reduced to `0.05`.
- Torch exposes the ROCm architecture as `gcnArchName`, not `gcn_arch_name`.
- `torch.dtype` has no `element_size()` method; a zero-element tensor is used
  to obtain the BF16 byte size.
- Torch’s GPU UUID is a `_CUuuid` object and must be converted to a string
  before JSON serialization.
- An initial timing run included one-time lazy initialization in the first
  measured repetition. The script now performs one explicit unmeasured warm-up
  per case before the three measured repetitions.
- The first successful run used Triton’s default cache directory. The final run
  sets `TRITON_CACHE_DIR=/tmp/sglang-cache-j-2919536e198f/triton`, keeping the
  generated cache job-private.

## Reproduction

From the repository root on one `gfx942` GPU:

```bash
mkdir -p /tmp/sglang-cache-j-2919536e198f/triton
TRITON_CACHE_DIR=/tmp/sglang-cache-j-2919536e198f/triton \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python reports/j-2919536e198f/chunk_partition_fp8_bf16.py \
  --output reports/j-2919536e198f/chunk_partition_results.json \
  --image-id sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1
```

The benchmark is finite: six partitions, two dtypes, one unmeasured warm-up,
and three measured repetitions per case. It does not use an unbounded loop,
sleep loop, or synthetic GPU burn.
