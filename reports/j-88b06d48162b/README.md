# MI300X SWA/compressor state fidelity report

## Scope

This is a bounded, local-state fidelity check for the unified-KV bit-exact prefix-reuse proposal in upstream issue 34562. It exercises synthetic SWA and compressor state tensors through the actual capture, host save, restore, and reuse paths on one assigned AMD Instinct MI300X (`gfx942`). It is not a full DeepSeek-V4 multi-layer or host-tier production qualification, and no model weights were downloaded.

## Candidate

- Upstream issue: `sgl-project/sglang` issue 34562
- Candidate PR: `sgl-project/sglang` PR 32214
- Tested candidate head: `221fccc128a6a4604b953fa7f106fe2cf14f987e`
- Candidate PR base: `03d06a764e4a83268eefd1bafc676418f7269c89`
- Delivery branch base: `amdpilot-org/sglang` `main`

The upstream issue and PR were read only. No upstream issue, PR, or comment was modified.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one `AMD Instinct MI300X`, `gfx942`, capability `(9, 4)`, GUID `47961`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Candidate source path: `/tmp/sglang-candidate-j-88b06d48162b/python/sglang`
- Torch native path: `/opt/venv/lib/python3.10/site-packages/torch`
- `sgl_kernel` native path: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`

## Reproduction

From a clone of `amdpilot-org/sglang` on branch `main`:

```bash
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch --no-tags upstream 221fccc128a6a4604b953fa7f106fe2cf14f987e
git worktree add --detach /tmp/sglang-candidate-j-88b06d48162b 221fccc128a6a4604b953fa7f106fe2cf14f987e

PYTHONPATH=/tmp/sglang-candidate-j-88b06d48162b/python \
  /opt/venv/bin/python reports/j-88b06d48162b/mi300x_swa_state_fidelity.py

PYTHONPATH=/tmp/sglang-candidate-j-88b06d48162b/python \
  /opt/venv/bin/python -m pytest -q \
  test/srt/mem_cache/test_swa_bitexact_capture.py \
  test/srt/mem_cache/test_swa_bitexact_capture_reuse.py \
  test/srt/mem_cache/test_swa_bitexact_compress_state.py \
  test/srt/mem_cache/test_swa_bitexact_compress_state_reuse.py \
  test/srt/mem_cache/test_swa_bitexact_hicache.py \
  test/srt/mem_cache/test_swa_state_loc_addressing.py
```

## Results

The MI300X harness covered:

- aligned prefix boundary with a partial tail
- short prefix boundary with a partial tail
- non-dividing SWA ring with a partial tail

For every case:

- restored SWA bytes matched the cold-state reference
- restored compressor state bytes matched the cold-state reference
- a deterministic attention continuation matched the cold-state reference exactly
- unrelated SWA slots were unchanged
- unrelated compressor state rows were unchanged

Raw results are in `raw-results.txt`. The candidate’s existing ownership, isolation, and reuse guard suites passed unchanged: `194 passed, 6 warnings, 2 subtests passed`.

## Limitations

- This is a synthetic, local-state fidelity check, not a full model or production qualification.
- The candidate PR is not merged here; this delivery records the tested candidate commit and evidence.
- No full model weights or additional framework stacks were downloaded.
