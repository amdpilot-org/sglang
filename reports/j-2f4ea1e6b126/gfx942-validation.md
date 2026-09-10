# gfx942 validation: in-place vs separate-output and repeated suffix stability

## Scope

- Upstream context: sgl-project/sglang issue 35096 (read-only).
- Prior mirror scope: amdpilot-org/sglang issue 203 and PR 240.
- This follow-up checks the flat DeepSeek-V4 RoPE path for:
  - in-place versus separate-allocation equivalence,
  - suffix-byte preservation across repeated updates,
  - forward and inverse rotation,
  - explicit and implicit positions,
  - FP32, FP16, and BF16.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, unique ID `0x78cd0e3e671b4299`, serial `692440003970`.
- Image requested: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Image verification: Docker CLI is unavailable in this job container, so the local image ID was not independently re-read.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Triton: `3.7.0`, `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- ROCm HIP: `7.2.26015-fc0010cf6a`, `/opt/rocm/bin/hipcc`.
- Target source: `/job/sglang/python/sglang/kernels/ops/attention/deepseek_v4_rope.py`.

## Installed-source baseline

- Source commit: `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Command: `/opt/venv/bin/python /tmp/sglang_baseline_first.py`.
- First GPU execution elapsed: `3.612770477309823` seconds.
- Independent reference: real-valued GPT-J rotation from `precompute_freqs_cis`.
- Numerical gate: absolute difference greater than `1e-4` counted as differing.
- Results:
  - `rope_dim=6`: max error `2.4748899936676025`, 2 differing cells.
  - `rope_dim=96`: max error `3.5097293853759766`, 544 differing cells.
- Conclusion: installed flat path mismatches the independent reference and masked per-token sibling.

## Candidate tested

- Mirror PR 240 head: `97c7578d8eac7ac9fd04faab4eb3fff690328d12`.
- Candidate change: pass real `ROPE_DIM` as constexpr, clamp padded loads, and mask padded stores.
- This candidate already fixes the prior suffix-column scope, so no duplicate code change is made here.

## New-scope probe on unmodified main

- Main commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Command: `/opt/venv/bin/python /tmp/sglang_main_new_scope_probe.py`.
- Case: `rope_dim=6`, shape `(4,8)`, FP32, implicit positions, forward rotation, 3 updates.
- Result:
  - Update 0: in-place vs separate max error `0.44011372327804565`, 64 suffix mutations.
  - Update 1: in-place vs separate max error `0.5335394144058228`, 64 suffix mutations.
  - Update 2: in-place vs separate max error `0.5301783680915833`, 64 suffix mutations.
- Elapsed: `1.1595031712204218` seconds.
- Conclusion: unmodified main fails the new property.

## Candidate matrix

- Command: `TRITON_CACHE_DIR=/tmp/sglang-cache-j-2f4ea1e6b126/triton TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-2f4ea1e6b126/inductor PYTHONPATH=/job/sglang/python /opt/venv/bin/python /tmp/sglang_repeated_update_matrix.py`.
- Timing method: `time.perf_counter` around the full bounded matrix, including `torch.cuda.synchronize`.
- Matrix:
  - widths: `{6, 64, 96, 128, 192}`,
  - shapes: `{(4,8), (2,4), (3,5)}`,
  - dtypes: `{FP32, FP16, BF16}`,
  - positions: `{implicit, explicit reversed}`,
  - rotation: `{forward, inverse}`,
  - 3 repeated updates per case.
- Total: 180 cases, 1080 kernel invocations, elapsed `18.530341519974172` seconds.
- Numerical gates:
  - FP32: `rtol=1e-5`, `atol=1e-5`.
  - FP16: `rtol=2e-3`, `atol=2e-3`.
  - BF16: `rtol=2e-2`, `atol=2e-2`.
  - Suffix and in-place versus separate-allocation equality: exact.
- Aggregate results:
  - All 180 cases pass.
  - Max absolute error versus independent reference: `0.0009765625`.
  - Differing cells versus independent reference: `0`.
  - Total suffix mutations: `0`.
  - In-place equals separate allocation in every update: `true`.
  - Max absolute in-place versus separate error: `0`.

## Unsupported boundaries and limitations

- The public API has no separate-output parameter; separate-allocation clone is the supported control used here.
- `compute-sanitizer` is unavailable in this image, so sanitizer-level OOB validation was not repeated.
- Only one MI300X (`gfx942`) was used.
- No full model weights or model-level evaluation were used.
- No upstream issue, PR, or comment was posted or modified.

## Artifacts

- Raw candidate matrix: `reports/j-2f4ea1e6b126/gfx942-repeated-update-results.json`.
- Installed-source baseline: `reports/j-2f4ea1e6b126/baseline-first.json`.
