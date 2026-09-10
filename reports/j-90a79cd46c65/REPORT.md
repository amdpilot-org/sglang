# QSA FP8-cache versus BF16-cache equivalence on MI300X

## Result

- Mirror `main` commit `0084030179bfba86bfeb6d43f7997d4076329d2c` reproduces the BF16-query/FP8-KV mismatch in the real Triton chunk-prefill kernel.
- The already-open mirror candidate PR 249, commit `7b0462f5415b203fb06e4b84e6a9dcda397ab5cb`, fixes that contract.
- On one AMD Instinct MI300X (`gfx942`), the additional finite adversarial property passes: dequantized FP8-cache attention is bit-identical to equivalent BF16-cache attention for both prefill and chunk-prefill.
- Both paths independently satisfy the unchanged reference gate `rtol=3e-2, atol=3e-2` against a separate float32 einsum/softmax reference.
- No production code was changed in this PR because the demonstrated mismatch is already fixed by open PR 249; duplicating that patch would repeat fulfilled work.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, capability `(9, 4)`, unique ID `0xf70f4ffec9a1c0e6`, serial `692440003945`
- Driver: `6.19.14.31400000`
- Python: `/opt/venv/bin/python` (`3.10.12`)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, ROCm/HIP `7.2.26015-fc0010cf6a`
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Checkout source: `/job/sglang/python/sglang`
- Native `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`

## Installed-source baseline

The first successful GPU execution used the preinstalled source at `/sgl-workspace/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/test/registered/attention/test_aiter_fp8_q_unified_attention.py
```

Result: 3 tests and 4 subtests passed in 48.028 seconds wall clock (pytest reported 43.61 seconds). This installed-source baseline is environment context only and is not evidence for the later mirror checkout.

## Base mismatch

From mirror `main`, with `PYTHONPATH=/job/sglang/python`:

```bash
/opt/venv/bin/python reports/j-90a79cd46c65/adversarial_probe.py \
  --mode base --output reports/j-90a79cd46c65/base-mismatch.json
```

The real Triton kernel rejected the mixed BF16×FP8 dot:

```text
CompilationError: Unsupported rhs dtype fp8e4nv
``+

The command completed in 6.633 seconds. This is the demonstrated mismatch; no synthetic burn or unbounded loop was used.

## Candidate property

The candidate was checked out at commit `7b0462f5415b203fb06e4b84e6a9dcda397ab5cb` and tested with:

```bash
export PYTHONPATH=/job/sglang/python

/opt/venv/bin/python reports/j-90a79cd46c65/adversarial_probe.py \
  --mode candidate --output reports/j-90a79cd46c65/adversarial-results.json

/opt/venv/bin/python -m pytest -q \
  test/registered/kernels/test_qsa_fp8_kv.py -v --tb=long

/opt/venv/bin/python -m pytest -q \
  test/registered/kernel/qsa/test_qsa.py --tb=short
```

The adversarial matrix is finite and deterministic. It tiles seven values—negative/positive maxima, `±1`, signed subnormal `±2^-9`, and zero—across BF16 queries and FP8 K/V. Explicit scales are `k_scale=0.25`, `v_scale=0.5`, and softmax scale `128^-0.5`. The BF16 cache is constructed from the exact dequantized FP8 values, isolating cache representation and scale handling rather than FP8 quantization error.

### Raw numerical results

- Prefill FP8-cache versus BF16-cache: maximum absolute error `0.0`, mean absolute error `0.0`, finite.
- Chunk-prefill FP8-cache versus BF16-cache: maximum absolute error `0.0`, mean absolute error `0.0`, finite.
- Prefill FP8-cache versus independent float32 reference: maximum absolute error `0.37732696533203125`, mean absolute error `0.010057314299046993`.
- Prefill BF16-cache versus independent float32 reference: maximum absolute error `0.37732696533203125`, mean absolute error `0.010057314299046993`.
- Chunk-prefill FP8-cache versus independent float32 reference: maximum absolute error `0.17140960693359375`, mean absolute error `0.007454427890479565`.
- Chunk-prefill BF16-cache versus independent float32 reference: maximum absolute error `0.17140960693359375`, mean absolute error `0.007454427890479565`.

The larger reference deltas occur only on the adversarial FP8 maximum magnitude and are identical for FP8 and BF16 paths, consistent with BF16 output rounding. The unchanged reference gate is `rtol=3e-2, atol=3e-2`; equivalence requires exact tensor equality.

### Bounded timing

After one warm-up call per path, each reported value is one synchronized `time.perf_counter` call:

- FP8 prefill: `0.22134929895401 ms`
- BF16 prefill: `0.11229701340198517 ms`
- FP8 chunk-prefill: `0.2688691020011902 ms`
- BF16 chunk-prefill: `0.17750076949596405 ms`

The complete candidate probe process took 16.934 seconds, including Python startup and four warm-up calls.

### Candidate suites

- `test/registered/kernels/test_qsa_fp8_kv.py`: 7 passed in 16.22 seconds; process elapsed 18.335 seconds.
- `test/registered/kernel/qsa/test_qsa.py`: 37 passed, 1 skipped in 19.78 seconds; process elapsed 24.452 seconds.

## Unsupported boundaries

- `_resolve_trtllm_sparse_decode()` returns `None` on gfx942 because its current gate targets SM100/SM120, not CDNA gfx942.
- The qualified stack has neither FA2 nor FA4, so `_resolve_flash_attn_varlen_func()` raises an `ImportError`.
- Therefore, this PR proves real Triton prefill and chunk-prefill equivalence, but not an end-to-end native decode kernel on gfx942.
- The candidate’s decode admission test uses a mocked TRT-LLM decode function; it is not an end-to-end decode proof.
- No model weights were downloaded, no node-wide state was modified, and no upstream issue, PR, or comment was posted.

## Scope

- Read-only upstream context: sgl-project/sglang issue 36545 and PRs 36556 and 36644.
- Existing mirror candidate: amdpilot-org/sglang PR 249.
- This report does not duplicate PR 249’s production patch. It records the separate adversarial FP8-dequantized versus BF16-cache equivalence property and confirms the existing fix preserves that contract.
