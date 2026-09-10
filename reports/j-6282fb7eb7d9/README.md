# gfx942 hidden-state gather and compaction evidence

## Scope

This report records a bounded, one-GPU validation of the existing speculative hidden-state movement helpers. It does **not** claim tensor-parallel execution, full-model correctness, throughput improvement, or verify-graph shape admission/acceptance behavior.

The tested callables were:

- `sglang.kernels.ops.speculative.gather_spec_extras.gather_spec_extras`
- `sglang.kernels.ops.speculative.dspark.dspark_verify_window.ScatterCompactToStrided.triton`
- `sglang.kernels.ops.speculative.dspark.dspark_verify_window.ScatterCompactToStrided.torch`

The persistent checkout was `amdpilot-org/sglang` at `0084030179bfba86bfeb6d43f7997d4076329d2c`. The preinstalled source used for the early baseline was `/sgl-workspace/sglang` at `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. The installed-source baseline is evidence for that revision only and is not proof for the later checkout.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, serial `692440004359`, node ID `3`, GUID `39656`.
- Image (operator-provided): `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`.
- Local image ID (operator-provided): `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Container hostname `banff-cyxtera-cx57-5` was not used as image identity.
- Python: `/opt/venv/bin/python` (`3.10.12`).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Torch native extension: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`.
- Triton: `3.7.0`, `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- Triton native library: `/opt/venv/lib/python3.10/site-packages/triton/_C/libtriton.so`.
- `sgl_kernel` package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- `sgl_kernel` native object: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- Test import side effect: AITER loaded `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.

## Commands and results

All timing used one cold process wall-clock measurement around the command. It includes Python import, Triton JIT/cache work, and GPU execution. No synthetic burn, unbounded loop, sleep loop, or repeated work was used.

### Installed-source early baseline

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_gather_spec_extras.py -s
```

- Result: `7 passed`, `19 subtests passed`.
- First successful GPU execution elapsed: `32.282853531 s`.
- Reference: exact advanced-index gathers with `rtol=0`, `atol=0`; source mutation and output alias checks.
- Coverage included repeated indices, empty indices, non-contiguous indices, index dtypes, hidden dtypes, and padded/non-power-of-two row widths.
- A first timing attempt using `/usr/bin/time` did not execute because `/usr/bin/time` is absent (`exit 127`). The successful timing used `date +%s%N`.
- Full record: `/job/baseline-first.json`.

### Persistent checkout gather gate

```bash
PYTHONPATH=/job/sglang/python \
PYTHONPYCACHEPREFIX=/tmp/sglang-cache-j-6282fb7eb7d9/pycache \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-6282fb7eb7d9/triton \
PYTHONNOUSERSITE=1 \
/opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/kernels/ops/speculative/test_gather_spec_extras.py -s
```

- Result: `7 passed`, `19 subtests passed`.
- Wall time: `19.169209301 s`.
- Imported `sglang` path: `/job/sglang/python/sglang/__init__.py`.

### Persistent checkout DSpark parity gate

```bash
PYTHONPATH=/job/sglang/python \
PYTHONPYCACHEPREFIX=/tmp/sglang-cache-j-6282fb7eb7d9/pycache \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-6282fb7eb7d9/triton \
PYTHONNOUSERSITE=1 \
timeout 420 /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/spec/dspark/test_dspark_kernel_parity.py::TestDsparkKernelParity::test_all_kernels_triton_matches_torch -s
```

- Result: `1 passed`, `22 subtests passed`.
- Wall time: `29.431036732 s`.
- The `scatter_compact_to_strided` case compares Triton against the independent Torch reference for exact and bucket-padded totals.

### Independent accepted-length, padding, repeated-index, and sentinel probe

The probe used `ScatterCompactToStrided.triton` and `ScatterCompactToStrided.torch` directly with:

- Accepted verify lengths: `[1, 3, 6, 2, 5, 1, 4, 6]`.
- Stride: `6`; accepted compact rows: `28`; padded compact rows: `48`.
- Hidden dimension: `333`, `bfloat16`.
- Sentinel: `-3.0`.
- Output shape: `[48, 333]`.

Results:

- Triton output exactly equaled the independent Torch reference.
- All 28 accepted rows exactly equaled their compact source rows.
- All 20 padded output rows exactly equaled `-3.0`; no accepted row was overwritten by the sentinel.

The same probe used `gather_spec_extras` with repeated indices `[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]`, truncated to 32 output rows, against a 12-row pool and 517-wide `bfloat16` hidden states:

- Minimum index `0`, maximum index `5`, both within the 12-row source bound.
- Every gathered output exactly equaled the corresponding advanced-index reference.
- Repeated source rows produced repeated exact copies.
- Probe wall time: `15.234191293 s`.

The first probe invocation had a harness-only mask-shape error (`IndexError`) before kernel comparison. The row mask was corrected to the flattened `[48]` output-row shape; the helper outputs were not implicated.

## Upstream context

Read-only context was taken from plain-text upstream issue `sgl-project/sglang` number `30734` and its comments. Related open upstream PRs reviewed were numbers `31047`, `31260`, `31457`, `32186`, and `32374`.

No upstream issue, PR, or comment was posted or changed.

The relevant helper and focused tests already exist in this mirror's `main`. Both existing gates and the independent probe pass on gfx942, so this report does not duplicate or alter a working fix. No production code change is included.
