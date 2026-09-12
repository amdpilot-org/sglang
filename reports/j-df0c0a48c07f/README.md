# Investigation: Qwen3.8 QSA with FP8 KV cache

The prepared base already contains the two open upstream candidate fixes relevant
to the report, so no additional runtime source change was justified.

- The original warmup/decode assertion is avoided on exact SM120 by
  `_resolve_trtllm_sparse_decode()`, which selects FlashInfer's TRT-LLM paged
  decode path. Its architecture boundary is regression-tested for SM120, SM100,
  and other SM12x devices.
- The separate cached chunk-prefill case converts gathered FP8 K/V values to
  the query dtype before Triton dot products. The checked-in regression exercises
  BF16 Q with FP8 K/V.
- `gpu_qsa_fp8_check.py` independently checks the latter computation against a
  PyTorch float32 softmax/einsum reference on the assigned gfx950 GPU, including
  top-k values 1, 16, and 17 and a BF16-cache control.

The original model weights and an NVIDIA SM120 GPU were unavailable. Therefore
this investigation does not claim a full server/model reproduction or direct
execution of the SM120 FA4/TRT-LLM routing fix. The upstream issue comments are
retained as problem/history evidence, not treated as local execution evidence.

Raw outputs and upstream metadata are stored beside this file.

