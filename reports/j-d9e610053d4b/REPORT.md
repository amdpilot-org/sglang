# Report: Paged GQA decode attention on AMD Instinct MI355X

## 1. The operation this repository is about

**Operation measured: paged GQA *flash-decoding* decode attention** — the
single-query (decode-step) attention over a *paged* KV cache with grouped-query
heads, i.e. the core of SGLang's **RadixAttention** decode path.

**Why this one.** SGLang is an LLM serving engine whose central primitive is
RadixAttention (cache-aware attention with prefix reuse). The latency-critical
cost of autoregressive serving is the per-token decode attention over the KV
cache, and SGLang ships exactly this kernel:

- `python/sglang/kernels/ops/attention/decode_attention.py` →
  `decode_attention_fwd`, a Triton flash-decoding kernel (page size 1,
  GQA, split-KV two-stage reduce), adapted from lightllm.
- `benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py` →
  the repo's own harness for that kernel.

The kernel consumes a paged KV store (`k/v_buffer`, `req_to_token` page table,
per-request `b_seq_len`) and produces one output token per request — the same
shape RadixAttention reuses a prefix cache for, and the dominant per-step cost in
serving. That makes it representative of the project, not incidental to it.

## 2. What I ran (and the one honest substitution)

The shipped `decode_attention_fwd` is a Triton kernel reached through the
`sglang` package. `sglang` is **not importable without a build** (no installed
wheel; `import sglang` fails), and the task says not to build the repo. `triton`
itself *is* installed (`3.5.1+rocm7.2`), but vendoring and re-wiring the kernel's
tuning/`score_mod` helpers out of the package was out of budget and risky.

**So, per the task's sanctioned path, I reimplemented the *operation* faithfully
in plain PyTorch** (`bench_paged_decode_attn.py`): gather each request's KV rows
out of the paged buffer via `req_to_token`, then run fused
`torch.nn.functional.scaled_dot_product_attention` (same math: QKᵀ·scale →
softmax → ·V). This is a **faithful reference, NOT the shipped fused Triton
kernel** — the absolute latencies below are therefore a *baseline* for the
operation on this GPU, not the production kernel's number. The shipped kernel
would be faster because it fuses the paged gather and avoids the head-permute
copy the reference performs (see "Gaps").

GQA (group=4) is handled by folding the group into the batch dimension and
broadcasting each KV head to its query heads with a stride-0 `expand` (no KV
materialization) — `enable_gqa=True` is **not honored** by the ROCm SDPA path in
this build, so it is expanded explicitly. A `--check` self-test compares the
implementation against a manual per-head reference on a tiny case
(`max_abs_diff=3.6e-7, allclose=True`).

## 3. Environment (read from the machine, not assumed)

| item | value |
|---|---|
| GPU | AMD Instinct MI355X (`gfx950`, device capability 9.5) |
| VRAM | 288 GiB (`309220868096` B total) |
| Compute units | 256 |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| torch HIP | `7.2.26015-fc0010cf6a` |
| ROCm (file) | `7.2.0` (`/opt/rocm/.info/version`) |
| hipcc | HIP `7.2.26015-fc0010cf6a`, clang `22.0.0git` |
| idle clocks | sclk 105–158 MHz, mclk 1900 MHz (GPU idle / low-power state) |

All of the above is captured at runtime by the script into `results.json`
(`env` block, including `rocm-smi --showproductname`, `--showclocks`,
`hipcc --version`, and the `gfx950`/`MI355X` lines from `rocminfo`).

Note: the SMI-exposed `mclk` (1900 MHz, read while idle) does **not** yield a
reliable peak-HBM figure for this part, so I report **achieved** bandwidth
below and deliberately do **not** assert a peak-vs-achieved ratio as if it were
measured.

## 4. Results

Workload: GQA, `num_q_heads=32`, `num_kv_heads=8` (group 4), `head_dim=128`,
bf16. Warmup 20, then 100 timed repeats per case (CUDA events, synchronized).
`kv_GB/s` = the operation's **intrinsic** KV-cache read traffic / wall time
(K+V over `[B, L, Hkv, D]`). `TFLOPS` = `4·B·Hq·L·D` / wall time.

```
   B      L    min_ms   mean_ms    med_ms    p99_ms   std_ms  spread%    kv_GB/s   TFLOPS
-----------------------------------------------------------------------------------------
   1   1024     0.080     0.085     0.083     0.103    0.010   116.67       52.6    0.211
   1   4096     0.197     0.202     0.200     0.216    0.007    33.52       85.2    0.341
   1  16384     0.659     0.667     0.667     0.687    0.005     6.74      101.8    0.407
   4   1024     0.083     0.088     0.087     0.107    0.004    37.78      202.9    0.812
   4   4096     0.225     0.229     0.228     0.241    0.007    29.68      297.6    1.190
   4  16384     0.864     0.872     0.871     0.886    0.005     3.98      310.8    1.243
  16   1024     0.122     0.126     0.125     0.137    0.004    17.54      552.1    2.208
  16   4096     0.557     0.566     0.565     0.584    0.005     6.97      481.7    1.927
  16  16384     2.167     2.183     2.182     2.207    0.011     4.22      495.4    1.982
  64   1024     0.430     0.437     0.435     0.454    0.006     8.92      623.9    2.495
  64   4096     2.176     2.192     2.191     2.210    0.008     1.65      493.5    1.974
  64  16384     8.682     8.754     8.755     8.810    0.028     1.92      494.7    1.979
 256   1024     1.643     1.654     1.653     1.670    0.006     1.79      653.7    2.615
 256   4096     8.741     8.864     8.866     8.939    0.041     2.37      491.4    1.966
 256  16384    34.461    34.605    34.605    34.732    0.061     1.26      498.5    1.994
```

### Reading the spread
- **Large shapes (B·L ≥ ~64·4k) are stable**: spread (max−min)/min ≈ 1–2%,
  std ≈ 0.01–0.06 ms. The number is trustworthy to ~±1%.
- **Small shapes (B·1·1024) are noisy**: B=1,L=1024 shows 117% spread because the
  kernel runs in ~80 µs — at/under event-timer and launch-jitter resolution.
  Treat those rows as "sub-100µs / timer-limited", not as a stable latency. They
  are retained for completeness; the stable signal is in the larger rows.

### Reading the bandwidth
- Intrinsic KV-read bandwidth plateaus at **~0.5 TB/s** for the large/long cases.
  This is a **lower bound on real HBM use**: the reference additionally performs
  a paged-gather copy and a head-permute copy of KV (to move heads adjacent to
  batch for GQA broadcasting), so the GPU touches several× the intrinsic KV
  traffic. Real HBM consumed is correspondingly higher; the fused shipped kernel
  removes those copies entirely. Decode attention is memory-bound, so the low
  TFLOPS (~2.0–2.6 at large batch) is expected, not a defect.

## 5. Reproduce

```sh
cd /tmp/delivery
python reports/j-d9e610053d4b/bench_paged_decode_attn.py \
  --batches 1,4,16,64,256 --seq-lens 1024,4096,16384 \
  --warmup 20 --repeats 100 \
  --json reports/j-d9e610053d4b/results.json
# correctness self-check (tiny case vs manual reference):
python reports/j-d9e610053d4b/bench_paged_decode_attn.py --check
```

Requires only PyTorch (built against ROCm) on the MI355X; no `sglang` build.
The printed env block and `results.json` are produced by the same invocation.

## 6. Gaps / what I did NOT do

- **Not the shipped kernel.** This measures a faithful PyTorch reference of the
  operation, not `decode_attention_fwd`. The shipped Triton kernel (fused paged
  gather, no permute copy, split-KV) would be materially faster; the numbers here
  are a baseline for the *operation*, not a production-kernel benchmark.
- **No peak-bandwidth claim.** SMI exposes only an idle `mclk`, so I report
  achieved bandwidth and explicitly do not state a peak-vs-achieved ratio.
- **MLA not measured.** SGLang's flagship AMD path for DeepSeek-class models is
  MLA attention (`hip_flash_mla.py`, `flashmla_backend`), a distinct variant. I
  measured standard GQA decode attention as the representative RadixAttention
  core; MLA is a related but separate operation left for a follow-up.
- **Static, page-size-1 mapping.** The page table is an identity mapping
  (contiguous cache) for a clean, reproducible measurement; real serving has
  fragmented pages, which adds indirection cost the fused kernel handles but
  this reference's gather would also incur.
- **GPU was in a low-power/idle clock state** during the run (per `rocm-smi`
  warning). Latencies are real for that state; a sustained-load clock state could
  differ. I did not lock clocks.
