# MI300X causal convolution cache-layout evidence

## Scope

This is the execution-representation follow-up for sgl-project/sglang issue 38622 and amdpilot-org/sglang issue 204. It does not repeat the original trigger: it tests the already-proposed upstream fix and extends coverage to the uncovered FP16/BF16 cache-layout representation.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one AMD Instinct MI300X, capability `(9, 4)`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, module `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, HIP `7.2.26015-fc0010cf6a`.
- Triton: `3.7.0`, module `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- Delivery base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Tested upstream candidate: sgl-project/sglang PR 38623 head `dac135a1957f72122b492c31655d808e124c3de8`.

## Installed-source baseline

Command: `/opt/venv/bin/python /job/baseline_first.py`.

The preinstalled source was commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, with `sglang` imported from `/sgl-workspace/sglang/python/sglang/__init__.py` and the Triton kernel from `/sgl-workspace/sglang/python/sglang/kernels/ops/mamba/causal_conv1d_triton.py`. This baseline is environment context only and is not proof for checkout changes.

The exact-state BF16/FP16 cases used an independent CPU float32 `torch.nn.functional.conv1d` reference, contiguous state storage followed by in-storage sentinels, and CUDA-event timing with two warmups and five measured calls. All sentinels remained unchanged. Width 5 was silently accepted by this pre-fix source.

Raw baseline data is in `baseline-first.json`.

## Candidate and delivery validation

The exact upstream candidate head was tested with:

```bash
PYTHONPATH=/tmp/sglang-pr38623/python /opt/venv/bin/python -m pytest -q \
  /tmp/sglang-pr38623/test/registered/layers/mamba/test_causal_conv1d.py::test_causal_conv1d_update_uses_tail_of_oversized_state \
  /tmp/sglang-pr38623/test/registered/layers/mamba/test_causal_conv1d.py::test_causal_conv1d_update_supports_circular_state
```

Result: 2 passed.

The delivery branch applies that candidate's two-file fix and adds the uncovered FP16/BF16 contract. Focused validation used:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/layers/mamba/test_causal_conv1d.py::test_causal_conv1d_update_uses_tail_of_oversized_state \
  /job/sglang/test/registered/layers/mamba/test_causal_conv1d.py::test_causal_conv1d_update_supports_circular_state \
  '/job/sglang/test/registered/layers/mamba/test_causal_conv1d.py::test_causal_conv1d_update_half_precision_cache_layout_contracts' \
  /job/sglang/test/registered/layers/mamba/test_causal_conv1d.py::test_causal_conv1d_update_rejects_unsupported_width
```

Result: 7 passed. The complete relevant file also passed:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  /job/sglang/test/registered/layers/mamba/test_causal_conv1d.py
```

Result: 252 passed.

## Bounded timing matrix

Each case used batch 2, width 4, two warmups, five measured CUDA-event calls, and an independent CPU float32 reference. Oversized state length was 8 with sequence length 3. Circular state length was 5 with sequence length 2 and cache sequence lengths `[0, 4]`.

| Dtype | Layout | Dim | Max output diff | Median |
|---|---|---:|---:|---:|
| FP16 | oversized | 128 | 0.00395918 | 0.121560 ms |
| FP16 | oversized | 2048 | 0.00604534 | 0.116068 ms |
| FP16 | circular | 128 | 0.00479794 | 0.122764 ms |
| FP16 | circular | 2048 | 0.00658989 | 0.120318 ms |
| BF16 | oversized | 128 | 0.03205180 | 0.113702 ms |
| BF16 | oversized | 2048 | 0.05121613 | 0.119075 ms |
| BF16 | circular | 128 | 0.04112816 | 0.117872 ms |
| BF16 | circular | 2048 | 0.04314351 | 0.116750 ms |

All eight cases preserved the state and storage addresses, retained the requested dtype, matched the independently represented half-precision state, and left sentinels unchanged. Width 5 failed before dispatch with `ValueError: causal_conv1d only supports width between 2 and 4, got 5`.

Raw delivery data is in `results.json`.

## Limitations

- No full model weights, toolchain replacement, node-wide state change, unbounded stress, or GPU burn was used.
- The installed-source baseline and checkout validation use the qualified Torch/ROCm stack; no alternate framework was installed.
- `ruff` is not installed in this image, so formatting was limited to `git diff --check` and the repository's existing style.
- No upstream issue, PR, or comment was posted or changed.
