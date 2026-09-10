# gfx942 ragged c128 plan validation

## Scope

This is a bounded, metadata-only validation of the DSV4 compress prefill planner. It does not run a full model, use multiple GPUs, or make a distributed-serving claim.

The investigation followed sgl-project/sglang issue 32470 and PR 32467. The final upstream formulation removes the redundant `warp_min` and `warp_max` initialization from `plan_compress_prefill_kernel0`. The tested mirror commit, `0084030179bfba86bfeb6d43f7997d4076329d2c`, contains that no-init formulation.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, unique ID `0x439e01ac3221d888`
- Driver: `6.19.14.31400000`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Assigned image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Assigned local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Image verification: the container has `/.dockerenv` and an overlay mount, but no Docker socket or image metadata API was available, so the assigned local image ID could not be independently queried from inside the container.
- Installed-source Python path: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Installed native kernel module: `/job/.cache/sglang/jit/gfx942/sgl_kernel_jit_dpsk_v4_compress_plan/build-f7a1ebb32ad81e88/deps-97d4b9841a6f0fe9/sgl_kernel_jit_dpsk_v4_compress_plan.so`
- Mirror Python path used for the final test: `/job/sglang/python/sglang/__init__.py`
- Mirror native kernel module: `/tmp/sglang-cache-j-1de017e2bc54/jit/gfx942/sgl_kernel_jit_dpsk_v4_compress_plan/build-f7a1ebb32ad81e88/deps-97d4b9841a6f0fe9/sgl_kernel_jit_dpsk_v4_compress_plan.so`

The installed-source baseline used commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, which already contained upstream fix commit `8549cce11b878d2fbf814d5e27bcc5e626890e70`. It is evidence for that installed source only, not for later checkout changes.

## Results

The bounded fixture used batch size 96, `extend_lens = [4] * 72 + [3] * 24`, compress ratio 128, ring size 256, and 360 query tokens. The independent eager reference expected valid ragged IDs to be exactly `0..359`.

Installed-source direct probe:

- Valid rows: `360 / 360`
- Maximum valid ragged ID: `359`
- GPU IDs matched the eager `arange(360)` reference
- GPU `plan_w` matched the CPU planner's `plan_w`
- Side-stream execution matched the eager reference
- `torch.cuda.CUDAGraph` capture succeeded; 32 replays matched the eager reference
- Timing method: `time.perf_counter` around 2,000 calls after 10 warmups, ending with `torch.cuda.synchronize`
- Timing: `0.08822489203885198` seconds, or `22669.339160191335` calls per second

The installed-source run of the pre-existing test file exercised the GPU path but reported `82 failed, 5 passed` because its Python decoder indexed a CPU `uint32` tensor, which Torch does not implement. The failure was in test decoding, not a planner assertion.

After fixing the decoder and adding the bounded ragged replay/stream test, the mirror run reported:

```text
6 passed, 3 warnings, 82 subtests passed in 20.20s
```

## Reproduction

Run from the repository root:

```bash
PYTHONPATH=/job/sglang/python \
SGLANG_CACHE_DIR=/tmp/sglang-cache-j-1de017e2bc54 \
/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/attention/test_deepseek_v4_compress_plan_draft_pad.py
```

The new test performs 256 direct planner calls, captures one planner graph, replays it 32 times, and checks one side-stream handoff. Every plan must contain exactly 360 valid IDs matching the eager reference.

## Limitations

- Results are specific to one MI300X (`gfx942`) with Torch ROCm 7.2 and the qualified installed stack.
- Graph capture and replay were supported on this stack, but that does not establish support on other ROCm versions or GPU architectures.
- The bounded timing measures planner calls only. It is not an end-to-end throughput benchmark.
- No NVIDIA B300/H20, multi-GPU, TP/DP, disaggregation, full-model, or distributed claim is made.
