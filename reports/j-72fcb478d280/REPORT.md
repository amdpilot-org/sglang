# gfx942 Qwen4 PLE investigation

## Scope and upstream context

- Read-only context: `sgl-project/sglang` issue 38731, the Qwen3.8-Flash-Next roadmap. It had no comments when read.
- Related candidate: `sgl-project/sglang` PR 38701, tested at commit `b254e1fd3a6f9f1c9ee48724c62c13168642efba`.
- No upstream issue, PR, comment, or review was posted or modified.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, 206,141,652,992 bytes of memory.
- Interpreter: `/opt/venv/bin/python` (Python 3.10).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- ROCm/HIP: `7.2.26015-fc0010cf6a`.
- Mirror PR base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Mirror source: `/job/sglang/python/sglang/kernels/ops/qwen4_ple.py`.
- Current native artifacts include `/job/.cache/sglang/triton/35NARWH3XTUCIBCF65UEGRRLL7MH7JRYB4LC6SCSHVXKB3J4373A/_qwen4_ngram_hash_kernel.hsaco` and `/job/.cache/sglang/triton/TFWKOVAZ4OTPWYCCNHMSVJ3D4DUTR7HJVJC4BVBWLF6OC7X5CTUQ/_gather_ple_embedding_from_pinned_kernel.hsaco`.
- Requested image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`. The container has no Docker/Podman socket, so the local image ID could not be independently queried; hostname was not treated as image identity.

## Installed-source baseline

- Installed source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Import path: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Kernel source: `/sgl-workspace/sglang/python/sglang/kernels/jit/csrc/ngram_embedding.cuh`.
- Native module: `/tmp/sglang-cache-j-72fcb478d280/gfx942/sgl_kernel_jit_ngram_embedding/build-4bc510426ba26483/deps-439295d6e91b6419/sgl_kernel_jit_ngram_embedding.so`.
- Command: `SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-72fcb478d280 SGLANG_HOME=/sgl-workspace/sglang /opt/venv/bin/python -m pytest -q /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_ngram_embedding.py`.
- Result: 10 passed in 32.43 seconds; first GPU execution elapsed time was 35 seconds. This is an installed-source baseline only and is not proof for later checkout changes.

## Finding and fix

The fused Qwen4 PLE Triton hash multiplied token IDs by large int64 multipliers. After wraparound, Triton's signed `%` can return a negative remainder on gfx942. The eager path uses `torch.remainder`, which normalizes the result into `[0, vocab_size)`. The fused path therefore produced different IDs for ordinary vocabulary-sized tokens.

The fix computes the signed remainder and adds `vocab_size` once when it is negative. This preserves the eager path's wrapped-int64 semantics without changing table scale, head offsets, TP ownership, or gather behavior.

## Validation

- New regression: `test_qwen4_fused_ngram_hash_matches_eager_remainder_after_int64_wrap`.
- Affected test command: `PYTHONPATH=/job/sglang/python SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-72fcb478d280 SGLANG_HOME=/job/sglang /opt/venv/bin/python -m pytest -q -p no:cacheprovider test/registered/kernel/embeddings/test_qwen4_ple_offload.py`.
- Result: 11 passed in 14.59 seconds.
- Synthetic control: `reports/j-72fcb478d280/gpu_validation.py`.
- Table semantics: 16 heads, 65,537 rows per head, TP size 8, and `ceil(16 * 65,537 / 8) = 131,074` local rows. This preserves the global table scale and rank-0 local range.
- Independent reference: CPU `torch.remainder` hash plus CPU row gather, with non-owner rows zeroed. All ten FP8 and BF16 cases had maximum absolute error 0.
- Warm timing: `triton.testing.do_bench_cudagraph(..., rep=20, return_mode="mean")`, which unrolls calls in a CUDA graph to avoid ROCm's fixed graph-replay overhead. Raw results are in `gpu_validation.json`.

## Architecture-specific limitations

- Only one MI300X (`gfx942`) was used; no multi-GPU or TP communication was run.
- The synthetic control is bounded to 1, 16, 64, 256, and 1024 tokens and does not download model weights or run an end-to-end model.
- Candidate PR 38701 is faster at 1, 16, and 64 tokens on gfx942, but slower at 256 and 1024 tokens in its single-rank repeated-key benchmark. Its fused gather is not duplicated here.
- The candidate's existing tests use small contexts and missed the int64-wrap remainder mismatch found by the independent reference.
- No node-wide state was changed. The ROCm NUMA-balancing warning was observed but not acted upon.
