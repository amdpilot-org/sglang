# gfx942 page-split offset control

## Scope and identities

- Campaign: `repo-e2e-20260909`; job: `j-c25addb40591`.
- Read-only upstream context: sgl-project/sglang issue 34025 and pull request 34027.
- Preserved upstream candidate commit: `2bb8e5db2ec7cf87a2f78a5e668aea42c4fca203`.
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c` (`main`).
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Interpreter: `/opt/venv/bin/python`; Torch `2.9.1+rocm7.2.0.git7e1940d4`; Triton `3.7.0`.
- Torch Python: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Torch native: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.
- Installed source: `/sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py` at `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Working source: `python/sglang/kernels/ops/attention/flash_mla_sm120.py`.
- GPU: one AMD Instinct MI300X, capability `(9, 4)` (`gfx942`), 206,141,652,992 bytes.

## Baseline and controls

The installed-source baseline ran before checkout changes. It split two pages with a 149,760-byte source stride and 37,440-byte destination stride, allocating 599,040 bytes. The data and scale regions matched an independent CPU byte reference. The first GPU execution took 859.0211109258235 ms wall time and 858.8712768554688 ms by CUDA event, including Triton JIT. This result is labeled installed-source evidence only and does not prove later checkout changes.

Mirror `main` already contains the u64-lane rewrite from upstream PR 29927. Its source and destination strides are 18,720 and 4,680 u64 lanes per page. A bounded index-only probe launched 458,868 programs and stored only the final four offsets; it did not allocate a large KV pool or dereference wrapped addresses. The int32 tail was `[2147483520, -2147474416, 2147483520, -2147469736]`; the independent int64 reference tail was `[2147483520, 2147492880, 2147483520, 2147497560]`. The int64 probe matched the reference.

After promoting `tl.program_id(0)` to int64, the same control matched the independent reference. A two-page real page-split remained byte-exact against the CPU reference, with 599,040 bytes allocated. The focused pytest run passed all three tests in 5.29 seconds on gfx942.

## Commands

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python /tmp/sglang-baseline-first.py
PYTHONPATH=/job/sglang/python /opt/venv/bin/python /tmp/sglang-mirror-main-control.py
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/attention/test_flash_mla_sm120_page_split.py
```

## Limitations

- MI300X is gfx942, not SM120. The production SM120 FlashMLA dispatch was not run end-to-end on this GPU.
- The boundary control reproduces the exact program-ID and lane-offset arithmetic but does not dereference the wrapped source or destination addresses.
- The two-page copy checks unchanged data and scale bytes, not a multi-million-token pool.
- Upstream PR 34027 is open and is not an ancestor of mirror `main`. Its byte-stride fix is superseded here by PR 29927's u64-lane form, so this change applies the same int64 promotion to the current lane arithmetic rather than duplicating the stale byte-path patch.
