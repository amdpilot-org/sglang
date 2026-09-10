# Reduced MI300X eager versus decode-graph replay study

## Result

This study exercised the real `sglang.srt.model_executor.model_runner.ModelRunner` forward path on one AMD Instinct MI300X (`gfx942`). It used a locally generated two-layer Llama-style configuration with dummy random in-memory weights, the Triton attention backend, and the full decode CUDA-graph backend. It did not use or substitute a plain Torch model as the SGLang engine.

All six workload cases passed the unchanged accuracy gate against an independent PyTorch Llama control. Every replay forward was admitted by the decode graph runner. Replay was faster than eager for batch sizes 1, 2, and 4, with complete-block mean speedups from 3.57x to 4.64x.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, unique ID `0x2e2f615a49e61496`, serial `692440004420`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Python: `/opt/venv/bin/python`, version 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP `7.2.26015-fc0010cf6a`.
- Tested mirror commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Tested Python source: `/job/sglang/python/sglang`.
- Native paths observed: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so` and `/opt/venv/lib/python3.10/site-packages/sglang_router/sglang_router_rs.abi3.so`.
- Job-private cache: `/tmp/sglang-cache-j-af9e61426adf`.

## Installed-source first GPU baseline

The first GPU execution used the preinstalled source at `/sgl-workspace/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, and its import path `/sgl-workspace/sglang/python/sglang`. This baseline is environment context only and is not evidence for later mirror-checkout changes.

The successful command was:

```bash
timeout 180s /opt/venv/bin/python \
  /sgl-workspace/sglang/test/registered/layers/test_layernorm_fusion.py \
  TestRMSNormInputShape.test_higher_rank_residual
```

It compared SGLang `RMSNorm` against `RMSNorm.forward_native` for a seeded `(2, 3, 512)` bfloat16 input and residual. The gate was `torch.testing.assert_close(atol=1e-2, rtol=1.5e-2)` for both outputs. The test passed with return code 0. Python `time.perf_counter` around one subprocess measured 35.902850034 seconds for first GPU execution; unittest reported 1.229 seconds for the test itself.

Two initial attempts are recorded in `/job/baseline-first.json`: `/usr/bin/time` was absent (return code 127), and the packaged unittest module path failed with `ModuleNotFoundError: No module named 'sglang.test.registered'` (return code 1). The direct source-file command above then succeeded.

## Model and workload

- Architecture: `LlamaForCausalLM`.
- Hidden size: 512; intermediate size: 1024; layers: 2.
- Attention heads: 8; KV heads: 2; head dimension: 64.
- Vocabulary: 512; maximum position embeddings: 128.
- Weight dtype: bfloat16; dummy-weight seed: 20260910.
- Dummy parameter bytes: 9,966,592, below the 4 GiB limit.
- Input length: 32 tokens per request.
- Batch sizes: 1, 2, and 4.
- Modes: eager and full decode-graph replay.
- Decode forwards per case: 23 total, comprising 3 warmups and 20 measured forwards.
- Workload cases: exactly 6, giving 138 decode forwards and 6 prefill forwards.
- No tokenizer, checkpoint, or model-weight download was used.

The independent control is a standalone PyTorch Llama forward implemented in `reduced_block_study.py`. It uses the loaded SGLang weights but does not call the SGLang model forward. It is a reference only, not an SGLang engine substitute.

## Measurements

The clock covers the complete decode block: input tensor creation, `batch.prepare_for_decode`, MLP-sync preparation when applicable, `ForwardBatch.init_new`, `ModelRunner.forward`, and `torch.cuda.synchronize`. It excludes request scheduling and sampling.

| Batch | Eager mean | Replay mean | Speedup | Replay median | Graph admissions |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.3089 ms | 0.6459 ms | 3.57x | 0.6440 ms | 23 / 23 |
| 2 | 2.7504 ms | 0.5926 ms | 4.64x | 0.5932 ms | 23 / 23 |
| 4 | 2.3418 ms | 0.6155 ms | 3.80x | 0.6125 ms | 23 / 23 |

The unchanged accuracy gate was cosine similarity at least 0.999, mean absolute error at most 0.02, and maximum absolute error at most 0.10. Actual replay cosine similarity was at least 0.9999972582; maximum absolute error was at most `1.574062480358407e-08`. Eager and replay final logits had maximum absolute difference 0.0 for every batch size.

Fresh input values were supplied on every replay. For example, batch 4 changed from `[325, 326, 327, 328]` on the first decode step to `[413, 414, 415, 416]` on the last. The final logits changed between the last two fresh inputs by `6.809830665588379e-06` maximum absolute difference for batch 4.

Across all 23 replays in each case, each static buffer field had exactly one stable data pointer: `input_ids`, `positions`, `seq_lens`, `seq_lens_cpu`, `out_cache_loc`, and `req_pool_indices`. This documents static-buffer reuse while fresh values are copied into those buffers.

Peak Torch allocated memory was 211,310,592 bytes and reserved memory was 228,589,568 bytes, both below the 48 GiB live-allocation limit. The study command had a 900-second timeout and completed in 8.81117296218872 seconds. The task wall limit was 7,200 seconds.

## Unsupported capture boundaries

The full decode runner falls back to eager at these boundaries:

- Batch size above the maximum captured bucket: `batch_size > max(capture_batch_sizes)`, which is 4 in this study.
- Token embedding overrides: `replace_embeds is not None`.
- Speculative width mismatch: `spec_info.num_tokens_per_req != captured_req_width`.
- Mixed encoder-decoder batch: an encoder-decoder batch containing `encoder_lens == 0`.
- Unsupported two-batch overlap: `forward_batch.can_run_tbo` is false when TBO is enabled.
- Ngram shape mismatch: `batch_size * captured_req_width != input_ids.numel`.

## Reproduction

From `/job/sglang`:

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_DISABLE_JIT=1
timeout 900s /opt/venv/bin/python \
  reports/j-af9e61426adf/reduced_block_study.py \
  --output reports/j-af9e61426adf/results.json
```

The command returns zero only when every accuracy gate passes and every replay forward is graph-admitted. Raw per-forward latency values, pointer sets, input samples, accuracy metrics, and memory measurements are in `results.json`.

## Upstream context and scope

Read-only context was taken from `sgl-project/sglang` issue 35003, “AMD Development Roadmap (2026 Q3).” At the time of this study it was open, had no comments, and linked broader platform, speculative-decoding, and context-parallelism work rather than a specific fix for this reduced-block replay comparison. No upstream issue, PR, or comment was posted or changed. No upstream candidate PR was tested because no already-working fix for this distinct scope was identified.

This is a reduced-block study, not a general performance claim. It uses one tiny random-weight model, one GPU, six cases, and no server or tokenizer. It does not merge or modify production behavior.
