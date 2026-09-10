# Triton decode execution-representation follow-up

## Scope

This follows up sgl-project/sglang issue 2271 without repeating its original long-context
split-K trigger. Upstream PR 2394 already merged the flash-decoding refactor, and upstream
PR 35801 remains a split-count tuning proposal. Neither covers packed K/V storage or
explicit rejection of unsupported last-dim strides.

The tested operation is `decode_attention_fwd_grouped` on one AMD Instinct MI300X
(gfx942). The packed case uses non-overlapping K and V views of one
`[tokens, kv_heads, 2 * head_dim]` bf16 tensor. Their slot and head strides are
non-contiguous, but each view has a contiguous last dim. The kernel already forwards those
strides, so this is a supported execution representation rather than a forced conversion.

## Results

- The packed K/V output matches an independent float32 PyTorch stable-softmax reference
  with `rtol=1e-2, atol=1e-2`; the observed maximum absolute error was `0.0009765625`.
- Packed and unpacked K/V use the same bf16 inputs and both match the reference.
- Sentinel regions before and after the output view remain unchanged.
- CUDA-graph capture and replay preserve the output tensor address and the reference result.
- Non-contiguous last-dim q and K views now fail with a clear `ValueError` instead of
  silently reading the wrong elements.
- Every benchmark case dispatches `_fwd_grouped_kernel_stage1` followed by
  `_fwd_kernel_stage2`.

The bounded timing matrix uses three warmups and ten timed calls per case. Raw results are
in `results.json`. At context lengths 128 and 1024, packed K/V measured within noise of
unpacked K/V on this small GQA shape; this is a representation check, not a general
performance claim.

## Reproduction

From the repository root with the qualified ROCm environment:

```bash
PYTHONPATH=python /opt/venv/bin/python -m pytest \
  test/registered/attention/test_triton_decode_representation.py -q

PYTHONPATH=python /opt/venv/bin/python \
  test/registered/attention/bench_triton_decode_representation.py \
  --output /tmp/representation-benchmark.json
```

The installed-source baseline was recorded separately before cloning. The gfx95-only
forced-splits test skipped on gfx942 with `the tuning only engages on gfx95`; the supported
neighboring control `TestTritonAttention::test_decode_attention` passed in 28.321833
seconds against its independent PyTorch reference.
