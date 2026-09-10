# gfx942 ModelRunner layer-chunk partition study

## Scope

This is a distinct chunk-partitioning follow-up to amdpilot-org/sglang issue 378. It does not repeat the completed DSpark row-partition study in mirror PR 474 or the sparse-MLA boundary study in mirror PR 412. Instead, it exercises the real SGLang `ModelRunner` and `LlamaForCausalLM.forward_split_prefill` path with one locally generated Llama-style configuration, dummy random weights, and six fixed legal layer partitions.

The read-only upstream context was sgl-project/sglang issue 35003. That issue is an AMD roadmap document with no comments. The relevant upstream split-prefill test repair, sgl-project/sglang PR 36617, is already merged at commit `e59a576f0301829ae15fd8c65b25ef819c335df6`; this study did not duplicate or modify that fix.

## Installed-source baseline

The first GPU attempt used the preinstalled interpreter and the existing RoPE kernel test at `/sgl-workspace/sglang/test/registered/kernels/ops/attention/test_pos_enc.py`. The installed JIT path failed before GPU execution because `hipcc` could not find `cuda_fp16.h` while compiling `sgl_kernel_jit_rotary_embedding` for `gfx942`.

The supported neighboring control used the prebuilt `sgl_kernel.rotary_embedding` operator on a `2 x 512`, 16-query-head, 4-KV-head bf16 case. It matched the independent Torch RoPE reference exactly, with one synchronized execution taking `0.001340024173259735 s`. First GPU execution, including setup and synchronization, took `2.857650407589972 s`. The complete artifact is `/job/baseline-first.json`.

This baseline is labeled installed-source evidence only. It is not proof for the persistent checkout used below.

## Persistent checkout

- Base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Branch: `amdpilot/j-c905c499f1be`
- Python: `/opt/venv/bin/python`
- SGLang source: `/job/sglang/python/sglang/__init__.py`
- ModelRunner source: `/job/sglang/python/sglang/srt/model_executor/model_runner.py`
- Native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- GPU: one AMD Instinct MI300X, capability `9.4` (`gfx942`)
- Image: operator-provided local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`

## Workload

The locally generated model has:

- `LlamaForCausalLM` architecture
- 8 hidden layers
- hidden size 512
- intermediate size 1024
- 8 attention heads and 2 KV heads
- vocabulary size 1024
- maximum position 4096
- bfloat16 weights
- seed 30734
- 18,883,072 parameters
- 37,774,336 weight-file bytes, well below the 4 GB limit

Every case uses the same seeded synthetic input IDs: batch size 2, input length 512, and 1,024 useful tokens. Each complete forward performs 8,192 useful layer-tokens. No tokenizer or checkpoint is downloaded.

The six cases are:

| Case | Layer partition |
|---|---:|
| `normal_unchunked` | normal `ModelRunner.forward` |
| `split_8` | `[8]` |
| `split_4_4` | `[4, 4]` |
| `split_2_2_2_2` | `[2, 2, 2, 2]` |
| `split_1x8` | `[1] x 8` |
| `split_5_3` | `[5, 3]` |

Each case records one cold complete forward and five warm complete forwards. The first cold normal forward includes first-use compilation/initialization. Timing uses `torch.cuda.synchronize` before and after each complete bounded forward and `time.perf_counter`. Memory uses `torch.cuda.reset_peak_memory_stats` before each case and records peak allocated and reserved bytes after that case's cold plus five warm forwards.

## Results

| Case | Warm median (ms) | Useful tokens/s | Useful layer-tokens/s | Peak allocated (GiB) | Peak reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| `normal_unchunked` | 9.661 | 105,991 | 847,927 | 9.819 | 9.844 |
| `split_8` | 9.292 | 110,206 | 881,649 | 9.821 | 9.846 |
| `split_4_4` | 9.492 | 107,878 | 863,025 | 9.821 | 9.846 |
| `split_2_2_2_2` | 9.753 | 104,997 | 839,974 | 9.821 | 9.846 |
| `split_1x8` | 9.018 | 113,546 | 908,372 | 9.821 | 9.846 |
| `split_5_3` | 9.437 | 108,512 | 868,094 | 9.821 | 9.846 |

All split cases finish with `split_index=8`. Their next-token logits and final full hidden states are bitwise identical to the normal unchunked SGLang result. The unchanged existing internal gate remains `atol=1e-3`, `rtol=1e-3`; no gate was changed.

The independent Torch control loads the same local weights with `local_files_only=True`. Compared with normal SGLang output, its last-token logits have maximum absolute error `6.103515625e-05` and mean absolute error `2.09808349609375e-05`. Its final hidden states have maximum absolute error `0.0003662109375` and mean absolute error `5.855155177414417e-05`. Both pass the separate cross-framework control gate `atol=2e-2`, `rtol=2e-2`.

The throughput differences are small and noisy at this reduced size; they should not be treated as a stable partition-ordering conclusion. The useful observation is that all legal partitions preserve output and final state while performing the same 8,192 layer-tokens of useful work.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-c905c499f1be/chunk_partition_model_runner.py \
  --warm-iters 5
```

The script generates the local model only if `/tmp/sglang-cache-j-c905c499f1be/llama-synthetic/model.safetensors` is absent. It does not download a tokenizer, checkpoint, or model weights. Raw results are in `reports/j-c905c499f1be/results.json`.

## Boundaries and unfinished work

- This is a reduced ModelRunner study, not an end-to-end serving benchmark.
- CUDA graphs are intentionally disabled for the `torch_native` attention backend.
- The installed JIT RoPE baseline is unsupported in this image because `cuda_fp16.h` is unavailable to `hipcc`; the prebuilt native operator remains supported.
- No upstream issue, PR, or comment was posted or changed.
- No production code change is proposed; this PR records the bounded investigation and its raw evidence.
