# gfx942 MoE dispatch investigation

This report records the bounded one-GPU investigation for sgl-project/sglang issue 36395. It is intentionally report-only: upstream pull request 36437 already contains the working fix, so this branch does not duplicate that code change.

## Environment identity

- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Container hostname (`banff-cyxtera-cx57-5`) is not used as image identity.
- GPU: one assigned AMD Instinct MI300X, device ID `0x74a1`, GUID `19304`, gfx version `gfx942`, 206 GiB visible.
- Interpreter: `/opt/venv/bin/python` resolves to `/usr/bin/python3.10`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, imported from `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`; HIP `7.2.26015-fc0010cf6a`.
- Installed SGLang: `0.5.18.dev20260826+g937af8538b`, imported from `/sgl-workspace/sglang/python/sglang`.
- Installed AITER: Python source `/sgl-workspace/aiter/aiter`; native modules `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`, `/sgl-workspace/aiter/aiter/jit/module_moe_sorting.so`, and gfx942 code object `/sgl-workspace/aiter/hsa/gfx942/fmoe_b16.co`.
- Delivery checkout: `/job/sglang`, branch `amdpilot/j-75f1f4c1dd25`, base commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.

## Installed-source first baseline

The first GPU execution objective used the preinstalled interpreter/toolchain and existing native AITER MoE kernel. This installed-source baseline is not evidence about later checkout changes.

1. Direct installed SGLang Triton fused-MoE failed before kernel dispatch with `ValueError: config namespace 'exec' not published`. A bare process must publish runtime configuration before this API can select a kernel config.
2. A direct AITER `fused_moe` control compiled `/sgl-workspace/aiter/aiter/jit/module_moe_ck2stages_b16_b16_preshuffle_off_b16_silu_no_mulWeightStage2.so` in 93.7 seconds, then failed with HIP error 700 (`illegal memory access`) on the tiny unshuffled g1u1 shape. Total first-attempt elapsed time was 106.914 seconds.
3. The supported neighboring control was AITER `fused_moe_bf16_asm.asm_moe`, a gfx942 BF16 assembly MoE using shuffled g1u0 weights. It completed on the assigned GPU.

The completed baseline used 8 experts, 64 tokens, hidden size 128, intermediate size 64, top-k 2, and BF16. It used 3 warmups and 20 timed calls with one synchronize after the loop:

- Mean wall time: 0.045632850378751755 ms/call; 20 calls took 0.0009126570075750351 seconds.
- Independent PyTorch reference: max absolute error 0.1328125, mean absolute error 0.026708455756306648, max relative error 1.0 (small-reference BF16 denominator).
- Profiler dispatch evidence: `fmoe_kernel_func` and the CK `MoeSortingKernel`; no CLI parsing was used.

The raw first-baseline artifact is `/job/baseline-first.json` outside the repository. Its values are also summarized in `gpu-evidence.json`.

## Dispatch proof

The dispatch probe publishes real `ServerArgs`, replaces only `ModelRunner` with a recorder, and reads `get_moe_runner_backend()` inside the fake runner constructor. This checks the process-global runtime value at the point where quantized MoE methods initialize, not CLI parsing.

On unpatched base commit `0084030179bfba86bfeb6d43f7997d4076329d2c`, `load_model` constructed all requested backends as `auto`:

| Requested | Backend at `ModelRunner` construction |
| --- | --- |
| `auto` | `auto` |
| `triton` | `auto` |
| `aiter` | `auto` |
| `flashinfer_mxfp4` | `auto` |

With the current-main adaptation of upstream PR 36437 (moving the three initializer calls from `latency_test` into `load_model`), construction observed the requested values:

| Requested | Backend at `ModelRunner` construction |
| --- | --- |
| `auto` | `auto` |
| `triton` | `triton` |
| `aiter` | `aiter` |
| `flashinfer_mxfp4` | `flashinfer_mxfp4` |

## Real numerical references

Both supported gfx942 runner cases were run after explicit runtime initialization and compared with independent per-token/per-expert PyTorch BF16 matmul/SiLU references. Timing used 3 warmups, 20 calls, and one synchronize; it is a small-kernel wall-time measurement, not a full-model benchmark.

| Backend | Profiler kernels | Max abs error | Mean abs error | Max rel error | Mean ms/call (base / adapted) |
| --- | --- | ---: | ---: | ---: | ---: |
| Triton | `fused_moe_kernel`, `moe_align_block_size_small_batch_expert_kernel`, `_moe_sum_reduce_kernel` | 0.000244140625 | 0.000015669269487261772 | 0.1220703050494194 | 0.19767375197261572 / 0.20916296634823084 |
| AITER | `fmoe_kernel_func`, CK `MoeSortingKernel` | 0.138671875 | 0.026375029236078262 | 1.0 | 0.05266615189611912 / 0.058055552653968334 |

The AITER maximum relative error is 1.0 because the independent reference has near-zero BF16 values; absolute error is the useful gate for that case. The Triton case used the default config because no tuned config exists for this tiny `E=8,N=64` MI300X shape, so its timing is not performance-optimal.

`flashinfer_mxfp4` initializes the runtime global correctly, but `Mxfp4MoEMethod` rejects this GPU with `NotImplementedError: moe_runner_backend=flashinfer_mxfp4 requires SM90, SM100, or SM120.` Therefore no truthful FlashInfer MXFP4 numerical reference is possible on MI300X gfx942.

## Upstream candidate

- Read-only issue context: sgl-project/sglang issue 36395, open, no comments at investigation time.
- Existing candidate: sgl-project/sglang pull request 36437, `[Bugfix] Initialize one_batch runtime configs in load_model`.
- Preserved tested candidate commit: `aaeaec16cf457d21c8023105cd1fbdaaaf917c5b` by gaoxiaomo, authored 2026-08-26 12:37:40 +0800.
- Candidate command: `PYTHONPATH=/tmp/pr36437/python /opt/venv/bin/python -m pytest -q test/registered/unit/bench/test_one_batch_runtime_config.py`.
- Candidate result: 1 passed, 3 warnings, 13.10 seconds.

The candidate moves MoE, FP8, and FP4 initialization ahead of `ModelRunner` construction and adds a regression test. Its base uses the older initializer API. The current mirror base already has the latency-path initializers and a published runtime context, so the temporary tested adaptation moved the current no-argument initializer calls into `load_model`. That adaptation produced the requested-backend table above. Because the open upstream PR is already the working fix, this branch reverts the temporary code adaptation and delivers this report instead of duplicating it.

## Reproduction

The dispatch/numerical harness is `moe_dispatch_gpu.py`. Run it against a checkout root:

```bash
cd /job/sglang
SGLANG_TEST_ROOT=/job/sglang \
SGLANG_TEST_COMMIT=$(git rev-parse HEAD) \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python reports/j-75f1f4c1dd25/moe_dispatch_gpu.py
```

The harness monkeypatches only the model-loading boundary needed to avoid downloading weights: `ModelConfig`, `ParallelState`, tokenizer, and `ModelRunner`. It does not fake the MoE backend global or the real Triton/AITER kernels. For the bare Triton kernel test it also disables the symmetric-memory allocation context because the small standalone process has no initialized tensor-parallel group.

No full model weights were downloaded, no node-wide state was modified, and no upstream issue, pull request, or comment was posted or changed.
