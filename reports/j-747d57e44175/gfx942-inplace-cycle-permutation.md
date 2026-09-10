# Bounded gfx942 in-place cycle permutation report

## Scope

This change adds a focused correctness probe for in-place KV-row cycle
permutations. It is not a production compaction kernel and does not integrate
with RadixCache or UnifiedRadixCache.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- GPU: one AMD Instinct MI300X, `gfx942`, target
  `amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-`
- Installed source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Installed native module:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Native module SHA-256:
  `d38b79daf2c5dd193d49837c903798e6ff5440cbf31d899c2fd8ca68fed6aa98`
- Persistent checkout: `/job/sglang`, base commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`

## Commands

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-747d57e44175/jit

/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_inplace_cycle_permutation.py

/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/kvcache/test_inplace_cycle_permutation.py \
  test/registered/kernels/ops/kvcache/test_hicache.py::test_hicache_transfer_mha
```

## Results

- Installed-source cache-move control: 8 passed in 65.83 seconds.
- Persistent-checkout cycle probe: 5 passed in 4.57 seconds.
- Persistent-checkout combined run: 13 passed in 48.34 seconds.
- Covered fixed points, two-cycles, a three-cycle, a four-cycle, and mixed
  cycles.
- Compared every active row against an independent out-of-place
  `index_select` reference.
- Preserved two unused trailing rows exactly.
- Used one scratch row and asserted both final allocator memory and peak
  allocator memory remained equal to the pre-call baseline.

## Limitations

- Cycle planning is host-driven; this probe does not demonstrate a production
  GPU kernel or the issue's throughput claims.
- The probe covers correctness and scratch bounds only, not latency or
  throughput.
- Validation is specific to one MI300X `gfx942`; NVIDIA Blackwell and other
  architectures were not tested.
- No full model weights or end-to-end cache eviction workload were used.
