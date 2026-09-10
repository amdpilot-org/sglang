# gfx942 Triton decode length/head scaling

## Scope

This report records a bounded, one-GPU investigation of Triton decode attention on AMD Instinct MI300X (`gfx942`). It keeps the operation contract and numerical gate unchanged:

```python
torch.allclose(output.float(), reference, atol=1e-2, rtol=1e-2)
```

The timing matrix uses batch 1, head dimension 64, MHA (`H_Q == H_KV`), and eight `(length, heads)` pairs whose product is 16384:

| Length | Heads |
|---:|---:|
| 16384 | 1 |
| 8192 | 2 |
| 4096 | 4 |
| 2048 | 8 |
| 1024 | 16 |
| 512 | 32 |
| 256 | 64 |
| 128 | 128 |

For MHA decode, estimated arithmetic intensity remains approximately 2 FLOP/byte across the matrix. Inputs are finite adversarial matrices with alternating `q` values of ±7, alternating `k` values of ±6, and periodic `v` values of ±2. The reference is an independently implemented float32 stable-softmax attention, not the kernel under test.

## Baseline

The installed source baseline used `/sgl-workspace/sglang` at commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`:

```bash
cd /sgl-workspace/sglang
/opt/venv/bin/python -m pytest -q \
  test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention \
  --durations=10
```

Result: `1 passed` in 48.99 seconds. A separate bounded timing case passed the same numerical gate with max absolute error `0.0` and median latency `0.107768 ms`.

## Current main

The persistent mirror checkout used `amdpilot-org/sglang` `main` at commit `0084030179bfba86bfeb6d43f7997d4076329d2c`. The current numerical test also passed:

```bash
cd /job/sglang
/opt/venv/bin/python -m pytest -q \
  test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention \
  --durations=10
```

Result: `1 passed` in 25.17 seconds.

### Timing matrix

Timing used CUDA events with 3 warmups and 10 timed calls; each row reports the median. All numerical gates passed.

| Length × heads | Max splits 8 | Max splits 16 | Max splits 64 |
|---:|---:|---:|---:|
| 16384 × 1 | 0.284415 ms | 0.168709 ms | 0.090769 ms |
| 8192 × 2 | 0.165902 ms | 0.108651 ms | 0.086118 ms |
| 4096 × 4 | 0.106245 ms | 0.077859 ms | 0.084676 ms |
| 2048 × 8 | 0.080626 ms | 0.078140 ms | 0.085517 ms |
| 1024 × 16 | 0.078060 ms | 0.079624 ms | 0.079143 ms |
| 512 × 32 | 0.078381 ms | 0.078541 ms | 0.078100 ms |
| 256 × 64 | 0.077098 ms | 0.078622 ms | 0.076096 ms |
| 128 × 128 | 0.077138 ms | 0.078100 ms | 0.075975 ms |

The actual HIP default is 16 splits, not the generic 8. At that default, latency still rises from about `0.078 ms` at short/high-head geometry to `0.169 ms` at long/low-head geometry, a 2.16× increase at fixed work and arithmetic intensity. Raising the explicit cap to 64 flattens the matrix to roughly `0.076–0.091 ms`.

## Candidate PRs

- Upstream PR 2394, commit `c35838da51bb4a75e2a4831cbb7e2d31088acc62`, already merged the original long-context flash-decoding optimization for issue 2271. This investigation did not repeat that fulfilled scope.
- Upstream PR 35801, commit `84a51946708edd03ddfc93a3a5b373d0408e7486`, was tested with the same matrix and max splits 8. All numerical gates passed. It reduced short-context split counts but did not improve the long/low-head cases; median latency remained `0.283934 ms` at 16384×1.
- Upstream PR 27786, commit `e6dca69353c1409b4fbb0346c954e897a89290e2`, was tested at the 16384 and 32768 length boundaries with max splits 64. Its 24576-token threshold kept 16384 at 8 splits and `0.312480 ms`; 32768 used 64 splits and `0.146176 ms`. Current `main` with explicit 64 splits measured `0.093375 ms` and `0.118594 ms` at those lengths.

## Conclusion

No production code was changed. The unchanged numerical contract passes on installed source, current `main`, and both tested candidate commits. The demonstrated mismatch is performance, not correctness: the HIP default of 16 splits leaves a 2.16× latency increase across the fixed-intensity matrix. An explicit 64-split cap removes most of that scaling, but promoting it was not justified by this bounded test because the intermediate `attn_logits` buffer scales with the cap and broader batch/head/model memory effects were not measured. PR 27786 already covers default-cap changes, and its 24576-token threshold leaves the 16384-token gfx942 boundary unsupported.

## Reproduction

From the repository root:

```bash
PYTHONPATH=python /opt/venv/bin/python \
  reports/j-df1c6dc74b95/benchmark.py
``+

The script prints the JSON matrix and uses the same independent reference, finite adversarial inputs, unchanged numerical gate, and bounded CUDA-event timing method.
