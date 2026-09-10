# MI300X SGLang backend comparison

## Scope

This is the `repo-e2e-20260909`, `rolling-feed-32f` backend-comparison follow-up. It is distinct from mirror issue 394, which studied eager versus captured block replay. This run compares two genuinely installed AMD execution backends on identical locally generated Llama-style weights and inputs:

- `triton`
- `aiter`

No tokenizer, checkpoint, or model weights were downloaded. No upstream issue, PR, or comment was modified.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- SGLang checkout: `/job/sglang`
- SGLang commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- SGLang import path: `/job/sglang/python/sglang/__init__.py`
- Triton import path: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Aiter import path: `/sgl-workspace/aiter/aiter/__init__.py`
- Torch import path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`

## Early installed-source baseline

Before cloning or editing, the installed SGLang activation kernel was run on the same GPU:

- Kernel: `sgl_kernel.silu_and_mul`
- Shape: `[2, 8, 8192]`
- Dtype: `torch.float16`
- Independent reference: `x[..., dim:] * torch.nn.functional.silu(x[..., :dim])`
- Numerical gate: `torch.testing.assert_close(rtol=1e-3, atol=1e-3)`
- Result: passed, max absolute difference `0.0`
- Timing: 20 CUDA-event iterations after 5 warmups, mean `0.014737950265407562 ms`
- First GPU execution elapsed time: `1.0748842051252723 s`
- Native extension: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`

The complete early baseline is saved at `/job/baseline-first.json`. It is explicitly labeled as installed-source evidence and is not proof for later checkout changes.

## Reduced model and weights

The benchmark uses a locally generated Llama-style configuration:

- Architecture: `LlamaForCausalLM`
- Hidden size: `512`
- Intermediate size: `1024`
- Layers: `4`
- Attention heads: `8`
- KV heads: `4`
- Head dimension: `64`
- Vocabulary size: `512`
- Context length: `512`
- Dtype: `float16`

Weights were generated once with SGLang’s dummy loader and saved locally:

- Path: `/tmp/sglang-cache-j-1e331eb0bb20/minimal-llama/model.safetensors`
- Size: `19,935,080` bytes
- SHA-256: `566682bbc82be148c43f38647228181c7c0bb34a4016b23956af91455a1b7d1d`
- Parameter count: `9,966,080`

Both backends loaded this identical weight file. No checkpoint download was used.

## Real SGLang forward paths

The benchmark uses SGLang’s existing offline `one_batch` path and a reduced `ModelRunner`:

- Triton backend class: `TritonAttnBackend`
- Triton backend source: `/job/sglang/python/sglang/srt/layers/attention/triton_backend.py`
- Aiter backend class: `AiterAttnBackend`
- Aiter backend source: `/job/sglang/python/sglang/srt/layers/attention/aiter_backend.py`

Profiler evidence confirms distinct native dispatch:

- Triton run includes SGLang’s Triton attention path and `create_flashinfer_kv_indices_triton`.
- Aiter run includes the CK Tile `FmhaBatchPrefillWithPagedKVCacheKernel` and Aiter’s `pa_ragged` native library.

Representative native paths:

- Triton package: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Aiter package: `/sgl-workspace/aiter/aiter/__init__.py`
- Aiter prefill kernel: `/tmp/sglang-cache-j-1e331eb0bb20/aiter-jit/mha_batch_prefill_fp16_nlogits_nbias_mask_nlse_ndropout_nqscale_nsink.so`
- Aiter paged-attention library: `/tmp/sglang-cache-j-1e331eb0bb20/aiter-root/build/pa_ragged_9c9b4ddee1b58c757eacc5d98975032f/lib.so`

## Workloads

Four bounded workload cases were used, below the six-case limit:

1. Correctness prefill: batch `1`, sequence length `32`
2. Prefill timing: batch `1`, sequence length `128`
3. Prefill timing: batch `4`, sequence length `64`
4. Decode timing: batch `4`, sequence length `64`

Each timing case used 5 warmup forwards and 20 measured forwards. Every iteration used fresh deterministic input values. No artificial burn, sleep loop, or unbounded repetition was used.

## Numerical control

An independent PyTorch Llama forward was implemented from the saved weights. Both SGLang backends were compared against it on the same deterministic input:

- Gate: `torch.testing.assert_close(rtol=5e-2, atol=5e-2)`
- Triton result: passed, max absolute difference `0.0`
- Aiter result: passed, max absolute difference `0.0`
- Cross-backend max absolute difference: `0.0`
- Cross-backend relative max difference: `0.0`

The independent PyTorch control is not labeled as a SGLang engine.

## Complete-block timing

Timing uses `time.perf_counter()` around the complete SGLang `extend` or `decode` block, with `torch.cuda.synchronize()` before and after. Results are the mean and median of 20 measured iterations after 5 warmups.

| Case | Triton mean | Triton median | Aiter mean | Aiter median |
|---|---:|---:|---:|---:|
| Prefill 1×128 | 3.906 ms | 3.891 ms | 3.606 ms | 3.569 ms |
| Prefill 4×64 | 4.481 ms | 4.058 ms | 3.590 ms | 3.568 ms |
| Decode 4×64 | 7.701 ms | 7.656 ms | 6.813 ms | 6.806 ms |

Peak live allocations stayed below 48 GB:

- Triton peak: `10,499,657,216` bytes
- Aiter peak: `10,516,977,664` bytes

## Commands

Run from `/job/sglang` with `/opt/venv/bin/python`:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-1e331eb0bb20/triton-cache
export AITER_ROOT_DIR=/tmp/sglang-cache-j-1e331eb0bb20/aiter-root
export AITER_JIT_DIR=/tmp/sglang-cache-j-1e331eb0bb20/aiter-jit

/opt/venv/bin/python reports/j-1e331eb0bb20/benchmark.py generate
/opt/venv/bin/python reports/j-1e331eb0bb20/benchmark.py triton
/opt/venv/bin/python reports/j-1e331eb0bb20/benchmark.py aiter
/opt/venv/bin/python reports/j-1e331eb0bb20/benchmark.py compare
```

## Boundaries and notes

- The installed-source early baseline is not evidence for later checkout changes.
- The first Aiter attempt used the default home cache under `/job/.aiter`; that cache was relocated outside `JOB_WORKDIR`, and the final Aiter run used `/tmp/sglang-cache-j-1e331eb0bb20/aiter-root` and `/tmp/sglang-cache-j-1e331eb0bb20/aiter-jit`.
- The first Triton run used the default cache under `/job/.triton`; that cache was relocated outside `JOB_WORKDIR`, and the final Triton run used `/tmp/sglang-cache-j-1e331eb0bb20/triton-cache`.
- No unsupported backend was relabeled as another backend.
- No CUDA graph or graph-replay comparison was performed; that is the distinct completed scope of mirror issue 394.
- No upstream issue, PR, or comment was posted or changed.
