# Long-context Triton decode-attention investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/2271

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3482

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Outcome

The historical missing-flash-decoding defect is not present in the current kernel. No
kernel source was changed. Current `decode_attention.py` already implements split-KV
flash decoding: stage 1 partitions each sequence into independently normalized partials,
and stage 2 merges their output/LSE pairs with a stable online-softmax reduction. The
Triton backend selects a per-request split count from context length, head geometry, and
device CU count.

On the assigned MI350X, the issue's Llama-3.1-8B decode shape (`B=1`, `Hq=32`, `Hkv=8`,
`D=128`, bf16) measured 60.7 us at context 200 and 68.4 us at context 2,000. It remained
68.2 us at 8,192 and reached 115.6 us at 32,768. A controlled run of the same current
kernel with `num_kv_splits=1` measured 126.6 us at 2,000, 520.8 us at 8,192, and 2,031.8
us at 32,768. Thus the existing split-KV path is both active and material; adding another
unmeasured implementation would duplicate an already effective optimization.

All ten direct cases matched an independent float32 PyTorch implementation at
`atol=0.02, rtol=0.02`. They include GQA and the separate MHA path, batch 1 and batch 8,
non-identity gathered KV indices, irregular lengths around 32/64-token boundaries,
heterogeneous batches, and masked tail lanes. Maximum absolute error across the cases was
0.00689.

## Reproduction

Run from the repository root with the prepared interpreter:

```bash
TRITON_CACHE_DIR=/tmp/amdpilot-repo-j-6d47a6865b8d/triton-cache \
PYTHONPATH=/job/repo/python \
/tmp/amdpilot-repo-j-6d47a6865b8d/venv/bin/python \
  reports/j-6d47a6865b8d/benchmark_decode_attention.py \
  --output reports/j-6d47a6865b8d/decode_attention_results.json --iters 50
```

The full JSON result and console log are retained beside this report. Focused repository
tests were also run:

```bash
TRITON_CACHE_DIR=/tmp/amdpilot-repo-j-6d47a6865b8d/triton-cache \
PYTHONPATH=/job/repo/python \
/tmp/amdpilot-repo-j-6d47a6865b8d/venv/bin/python -m pytest -q \
  test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention \
  test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_grouped_decode_attention
```

This passed `2 passed` on the assigned GPU. `compiler_artifacts.txt` records the private
Triton cache paths, SHA-256 hashes, and headers from the generated gfx950 AMDGCN for both
the grouped stage-1 and stage-2 kernels. The raw compiler cache remains outside the work
tree at `/tmp/amdpilot-repo-j-6d47a6865b8d/triton-cache`.

## Limitations

This is a direct kernel investigation, not an end-to-end serving comparison. The gated
Meta Llama weights were not available or needed, and the deterministic tiny-Llama
transport fixture cannot qualify Llama-3.1-8B performance. FlashInfer was not installed
or used, and the 2024 report's original NVIDIA hardware was not available. The newer
issue comment about Qwen3.5/speculative decoding concerns different model and serving
shapes and is not claimed resolved by these Llama-shaped direct-kernel measurements.
