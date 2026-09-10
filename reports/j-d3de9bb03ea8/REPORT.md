# gfx942 Qwen4 PLE output-reuse validation

## Scope and evidence

- This follows the PLE/embedding-gather context in amdpilot-org/sglang issue 235 without repeating its original signed-remainder trigger. The already-open mirror PR 325 covers that fix, and mirror PR 362 already reports index-property controls.
- Read-only context was taken from sgl-project/sglang issue 38731 (a roadmap issue with no comments) and sgl-project/sglang PR 38701 at commit `b254e1fd3a6f9f1c9ee48724c62c13168642efba`. No upstream issue, PR, or comment was posted or changed.
- The uncovered case here is fresh output allocation versus documented safe `out=` reuse across two distinct deterministic input batches, with NaN sentinel protection and an independent CPU row-index reference.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`.
- Image requested by the operator: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`. No Docker/Podman socket was available for independent image inspection.
- Interpreter: `/opt/venv/bin/python`; Torch `2.9.1+rocm7.2.0.git7e1940d4`; Triton and Torch paths are recorded in `gpu_reuse_results.json`.
- Mirror base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Source paths: `python/sglang/srt/models/qwen4_exp.py` and `test/registered/kernel/embeddings/test_qwen4_ple_offload.py`.
- Native dispatch observed by Torch profiler: `_gather_ple_embedding_from_pinned_kernel`. Job-private Triton `hsaco` paths are recorded in `gpu_reuse_results.json`.

## Installed-source first baseline

The installed source at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` did not contain the newer Qwen4 PLE files. Before cloning, the supported neighboring ngram embedding control was run:

```bash
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-d3de9bb03ea8 \
SGLANG_HOME=/sgl-workspace/sglang \
/opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  /sgl-workspace/sglang/test/registered/kernels/ops/speculative/test_ngram_embedding.py
```

Result: `10 passed, 1 warning in 30.04s`; first GPU-process wall time was `31.911272 s`. The test compares decode and general ngram paths with `atol=0, rtol=0`. This installed-source baseline is not proof for the later mirror checkout and is recorded in `/job/baseline-first.json`.

## Mirror validation

The focused suite was run from the mirror checkout:

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-d3de9bb03ea8
export SGLANG_HOME=/job/sglang
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-d3de9bb03ea8/triton
/opt/venv/bin/python -m pytest -q -p no:cacheprovider \
  test/registered/kernel/embeddings/test_qwen4_ple_offload.py
```

Result: `13 passed, 3 warnings in 15.72s`. The unchanged graph-buffer lifecycle test continues to require the same address for the same captured token count and distinct addresses for different token counts.

The bounded report harness ran 18 cases: BF16 and FP8-e4m3fn table storage, embedding dimensions 7/64/257, and token counts 1/16/64. Each mode used two warmups and ten timed calls, alternating two distinct input batches. CUDA events bracketed only the gather; fresh mode includes allocation, while reuse mode performs its sentinel fill outside the timed region.

| Case | Fresh mean (ms) | Reuse mean (ms) |
|---|---:|---:|
| BF16, dim 7, 1 token | 0.041 | 0.040 |
| BF16, dim 7, 16 tokens | 0.040 | 0.030 |
| BF16, dim 7, 64 tokens | 0.039 | 0.030 |
| BF16, dim 64, 1 token | 0.040 | 0.031 |
| BF16, dim 64, 16 tokens | 0.038 | 0.030 |
| BF16, dim 64, 64 tokens | 0.039 | 0.030 |
| BF16, dim 257, 1 token | 0.039 | 0.031 |
| BF16, dim 257, 16 tokens | 0.038 | 0.030 |
| BF16, dim 257, 64 tokens | 0.058 | 0.046 |
| FP8-e4m3fn, dim 7, 1 token | 0.039 | 0.030 |
| FP8-e4m3fn, dim 7, 16 tokens | 0.039 | 0.033 |
| FP8-e4m3fn, dim 7, 64 tokens | 0.040 | 0.032 |
| FP8-e4m3fn, dim 64, 1 token | 0.039 | 0.030 |
| FP8-e4m3fn, dim 64, 16 tokens | 0.038 | 0.029 |
| FP8-e4m3fn, dim 64, 64 tokens | 0.041 | 0.030 |
| FP8-e4m3fn, dim 257, 1 token | 0.039 | 0.030 |
| FP8-e4m3fn, dim 257, 16 tokens | 0.039 | 0.030 |
| FP8-e4m3fn, dim 257, 64 tokens | 0.057 | 0.045 |

## Result

- Every fresh and reused output matched the independent CPU reference exactly: maximum absolute error `0`, mismatch count `0`.
- Reuse returned the supplied output pointer, fully overwrote the NaN sentinel, and was safe across both distinct input batches for both storage dtypes.
- Fresh outputs did not alias the input IDs. Output dtype remained BF16 for both BF16 and FP8-e4m3fn table storage.
- Wrong output shape, dtype, and device were rejected with `ValueError`; no unsupported representation was forced through.
- No production gather behavior was changed. This is honest positive evidence for the documented reuse contract, not a claimed performance fix.

## Limitations

- Validation used one gfx942 GPU and did not run multi-GPU TP communication, a full model, CUDA Graph replay in the timing harness, or checkpoint downloads.
- The timing matrix is intentionally small and includes launch overhead; it should not be interpreted as an end-to-end model benchmark.
- The installed source and mirror checkout are different revisions, so the first baseline only establishes that the qualified stack and neighboring native control worked before checkout changes.
