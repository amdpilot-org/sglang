# gfx942 investigation: Qwen3.5 multimodal mRoPE

## Result

Issue 35345 is already fixed on `main` by upstream PR 34446 (commit
`e635577431cbdfb8ce5fafb0fcd8a4ac074062c6`). Open upstream PR 35744 (commit
`9b2e053ce0d203b368e68e915667295aea24df32`) is a later, duplicate candidate and
was not copied.

This change adds GPU coverage for the path actually used on ROCm:
`MRotaryEmbedding.forward_cuda` with `[3, T]` positions. The test compares the
Triton result with an independent per-axis `torch.gather` reference for
interleaved `[11, 11, 10]` and sectioned `[12, 10, 10]` layouts. A 1D control
uses the same reference.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`
- Torch path: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Installed `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`
- Delivery checkout: `/job/sglang`, branch `amdpilot/j-bfc69cff8dfa`
- PR base: `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Installed source context: `/sgl-workspace/sglang` at `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`
- Job-private caches: `/tmp/sglang-cache-j-bfc69cff8dfa`
- No model weights were downloaded.

## Reproduction

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-bfc69cff8dfa/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-bfc69cff8dfa/inductor
export TMPDIR=/tmp/sglang-cache-j-bfc69cff8dfa/tmp
/opt/venv/bin/python -m pytest \
  test/registered/rotary/test_mrope_forward_cuda.py -vv --no-header
```

Observed result: `2 passed, 1 warning in 7.59s`.

The numerical gate is unchanged at `atol=2e-2, rtol=2e-2`. In the standalone
probe recorded in `/job/baseline-first.json`, maximum absolute error was
`0.015625` for both Q and K in the 1D control and the `[3, 33]` interleaved
mRoPE case. Mean absolute errors were between `6.72e-5` and `8.25e-5`.

The probe used one `torch.cuda.Event` pair around each call. The first 1D call
took `22.06 ms`; the first 3D call took `903.15 ms` because it included Triton
JIT compilation. No synthetic GPU work or unbounded loops were used.

## Architecture limitation

The direct `fused_qk_gemma_rmsnorm_rope_gate` test is not supported by this
Triton/ROCm stack on gfx942. Both mirror `main` and upstream PR 35744 fail
during HSACO linking with:

```text
ld.lld: error: target emulation unknown: -m or at least one .o file required
```

PR 35744 also emits an invalid `griddepcontrol` instruction for gfx942. This is
an architecture/toolchain limitation, not a numerical mismatch. The production
Qwen3.5 ROCm dispatch does not use that CUDA-only fused path; it uses
`forward_prepare_fused_gate` followed by `MRotaryEmbedding`, which passes the
independent per-axis reference on MI300X.

The installed-source baseline and raw command outputs are preserved in
`/job/baseline-first.json`. That baseline describes the preinstalled source and
is not evidence about later checkout changes.
