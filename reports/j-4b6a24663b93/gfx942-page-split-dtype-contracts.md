# gfx942 page-split dtype and reuse contracts

## Scope

This is a distinct execution-representation and reuse follow-up to amdpilot-org/sglang issue 220. It does not repeat or duplicate the int32 lane-offset fix in mirror pull request 248. The uncovered case here is the raw-byte dtype contract of `_split_kv_pages_to_64`: its parameter and stride arithmetic are byte-based, but the function did not reject non-`uint8` tensors.

The experiment covers the documented caller path, where FP16, BF16, and FP8 KV storage is first viewed as `uint8`, and compares the copied native values against an independently constructed destination. It also covers the documented `src_pbs == 64` alias and the persistent destination address used by CUDA graph replay. Unsupported direct non-`uint8` inputs now fail with a clear assertion instead of being forced through.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Interpreter: `/opt/venv/bin/python`, Python `3.10.12`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, path `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Triton: `3.7.0`, path `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- Installed-source `sglang`: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Installed `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- Checkout source: `/job/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py`.
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, serial `692440003949`, node ID 8, GUID 25408.
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c` (`main`).
- Job-private Triton cache: `/tmp/sglang-cache-j-4b6a24663b93/triton`.

## Installed-source baseline

The first GPU objective used the preinstalled source before cloning or editing. The relevant installed test is architecture-gated:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/kernels/ops/attention/test_flash_mla_backends.py::TestTouchedPageSplit::test_only_marked_pages_are_split
```

Result: `1 skipped`; the class requires compute capability `(12, 0)`, while the assigned GPU is `gfx942`.

The supported neighboring control then invoked the installed `_page_split_kernel` through `_split_kv_pages_to_64` directly. It used three pages, a sentinel-filled persistent destination, touched indices `[0, 5, 515, -1]`, and an independent CPU byte-copy reference. The copied regions matched exactly, all untouched destination bytes and alignment padding retained the `0xA5` sentinel, and the first GPU execution completed in `20.656 s` from process start (`0.882 s` for the first call, including Triton JIT).

Full installed-source paths, commands, native cache key, and the first bounded timing matrix are in `/job/baseline-first.json`. That baseline is environment context only and is not proof for checkout changes.

## Pre-change checkout evidence

Before adding the dtype gate, the checkout control built finite FP16, BF16, and FP8 values in the documented flat page layout and viewed them as `uint8` before calling `_split_kv_pages_to_64`. All three documented paths were byte-exact and value-exact against the independent destination reference, with no sentinel violations.

Direct calls that bypassed the documented `uint8` view did not fail clearly:

| Direct input | Pre-change result |
| --- | --- |
| FP16 | Silently copied wrong bytes; reference mismatch |
| BF16 | Silently copied wrong bytes; reference mismatch |
| FP8 | Silently produced byte-exact output despite violating the raw-byte contract |

The FP16 and BF16 failures follow from element strides being interpreted as byte strides. FP8 happens to be one byte per element, but it is still outside the documented `uint8` interface and must not be forced through.

## Change

- `_split_kv_pages_to_64` now asserts `kv_u8.dtype == torch.uint8` before alias handling or buffer allocation.
- The assertion names the required caller conversion explicitly.
- No numerical operation, layout, aliasing behavior, persistent-buffer key, or CUDA graph address behavior was changed.

## Validation

Focused checkout test:

```bash
PYTHONPATH=/job/sglang/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-4b6a24663b93/triton \
/opt/venv/bin/python -m pytest -q -s \
  /job/sglang/test/registered/kernels/ops/attention/test_flash_mla_sm120_page_split_contracts.py
```

Result: `4 passed`, `6 subtests passed`, in `5.65 s`. The tests cover:

- FP16, BF16, and FP8 `uint8` views against independent byte and native-dtype references.
- Sentinel-protected destination padding and untouched storage.
- Clear assertion failures for direct FP16, BF16, and FP8 inputs.
- The documented `src_pbs == 64` input alias.
- CUDA graph capture and replay with an unchanged persistent destination address.

Adjacent production test:

```bash
PYTHONPATH=/job/sglang/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-4b6a24663b93/triton \
/opt/venv/bin/python -m pytest -q -s \
  test/registered/kernels/ops/attention/test_flash_mla_backends.py::TestTouchedPageSplit::test_only_marked_pages_are_split
```

Result: `1 skipped` in `11.89 s`, as expected on `gfx942` because the production SM120 test requires compute capability `(12, 0)`.

Static checks: `git diff --check` passed and both changed Python files compiled with `py_compile`. `ruff` is not installed in the qualified environment, so formatting could not be run.

## Bounded timing matrix

Timing used CUDA events with two warmups and five timed calls per cell; the median is reported. The matrix is deliberately limited to 1, 4, and 16 pages with and without a one-page touched mask.

| Pages | Mask | Median |
| ---: | --- | ---: |
| 1 | no | `0.035281 ms` |
| 1 | yes | `0.066754 ms` |
| 4 | no | `0.036725 ms` |
| 4 | yes | `0.066674 ms` |
| 16 | no | `0.035482 ms` |
| 16 | yes | `0.071285 ms` |

Raw samples, the timing method, GPU identity, Triton cache key `e04edf1f7fcdf02296abf59c88f071e1341c8570e506f0f6c69ab03c4310638e`, and actual `.hsaco` paths are recorded in `gfx942-page-split-dtype-contracts.json`.

## Limitations

- The assigned GPU is MI300X/`gfx942`, not SM120, so the production SM120 FlashMLA dispatcher was not run end to end.
- The direct Triton page-split kernel is architecture-generic and was executed on the assigned GPU; this is meaningful neighboring evidence, not an SM120 end-to-end proof.
- No full model weights, multi-million-token KV pool, invalid wrapped-address dereference, toolchain replacement, unbounded stress, or node-wide state change was used.
- No upstream issue, pull request, or comment was posted or modified.
