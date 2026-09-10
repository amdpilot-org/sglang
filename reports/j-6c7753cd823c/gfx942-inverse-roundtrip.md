# gfx942 inverse permutation roundtrip report

## Scope

This follow-up tests whether an in-place KV-row permutation followed by its
inverse restores every original cache byte. It is separate from the forward
permutation scope already covered by mirror issue 210 and PR 288.

No implementation change was made because the tested operation contract did not
demonstrate a mismatch.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- GPU: one AMD Instinct MI300X, `gfx942`, unique ID
  `0x439e01ac3221d888`, serial `692440003981`
- Installed source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed native module:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Native module SHA-256:
  `d38b79daf2c5dd193d49837c903798e6ff5440cbf31d899c2fd8ca68fed6aa98`
- Persistent checkout: `/job/sglang`, base commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Installed-source baseline

The first GPU execution used the existing installed HiCache MHA transfer test:

```bash
PYTHONPATH=/sgl-workspace/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-6c7753cd823c/installed-jit \
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/kvcache/test_hicache.py::test_hicache_transfer_mha
```

Result:

- 8 passed.
- Pytest duration: 63.80 seconds.
- First GPU execution wall time: 66.832 seconds, measured with `date +%s%N`
  immediately before and after the pytest process.
- The test compares backed-up and reloaded K/V pages with `torch.equal` and
  checks untouched host pages remain unchanged.
- This baseline is installed-source evidence only and is not proof for the
  persistent checkout.

An initial attempt to use `/usr/bin/time` failed because that binary is absent;
it did not execute GPU work.

## Inverse-roundtrip probe

The probe preserved the operation contract from the prior forward-permutation
work:

- `output[i] = input[permutation[i]]`
- one scratch row
- unused trailing rows remain unchanged
- final and peak CUDA allocator memory remain equal to the pre-call baseline

The inverse was derived independently with
`inverse[permutation[i]] = i` and cross-checked against `torch.argsort`.
Forward and inverse results were compared with independent out-of-place
`index_select` references. The final tensor was also compared byte-for-byte
with a clone of the original tensor.

The finite adversarial matrix used `bfloat16` rows viewed as `uint8`. It included
every byte value from 0 through 255 and explicit bit patterns for signed zero,
positive and negative infinity, quiet and signaling NaNs, maximum finite,
minimum normal, minimum subnormal, and representative finite values. Two
unused trailing rows were retained.

### Cases

| Case | Cycle lengths | Forward event (ms) | Inverse event (ms) | Total event (ms) | Wall (ms) |
|---|---:|---:|---:|---:|---:|
| fixed points | 1, 1, 1, 1 | 0.096181 | 0.067355 | 0.163536 | 0.318391 |
| two cycles | 2, 2 | 0.062905 | 0.055247 | 0.118152 | 0.211502 |
| three cycle | 3 | 0.039570 | 0.039250 | 0.078821 | 0.151743 |
| four cycle | 4 | 0.051038 | 0.048392 | 0.099430 | 0.185450 |
| mixed cycle lengths | 1, 2, 3, 4, 1 | 0.162856 | 0.167867 | 0.330723 | 0.470966 |

All five cases passed:

- forward bytes matched the independent `index_select` reference
- inverse bytes matched the independent inverse reference
- every original byte was restored
- unused rows remained byte-identical
- final and peak allocator memory matched the pre-call baseline

Timing used one CUDA event pair around each direction and one wall-clock
measurement around the complete probe. Each case ran once.

## Evidence review

- Upstream issue 38357 was read only; it has no comments and no timeline
  entries at the time of inspection.
- An exact upstream PR search for `38357` returned no results.
- Mirror issue 210 and PR 288 already cover the forward permutation scope,
  including fixed points, two-cycles, a three-cycle, a four-cycle, mixed
  cycles, one scratch row, and allocator bounds.
- This follow-up did not repeat that fulfilled scope; it tested only the
  additional inverse-restoration property.

## Limitations

- Validation covers one MI300X `gfx942` only.
- Cycle planning is host-driven; this is not a production compaction kernel.
- No latency or throughput claim is made.
- No full model weights or end-to-end cache eviction workload were used.
- GPU `torch.equal` calls can temporarily increase peak allocator memory by
  about 1 KiB, so allocator snapshots were taken immediately after operation
  synchronization and before reference comparisons.
- No unsupported boundary was encountered for the tested finite matrix.
