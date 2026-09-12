# Independent review of PR 1452

Reviewed exact candidate commit `dec541903732fd188d1e4c22d7790fb99571c10d`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate is a full fix for the original
kernel contract. On the base, its regression failed 5 of 15 cases and direct
sentinel checks proved writes past `rope_dim` for widths 6, 96, and 192. At the
candidate, all 15 regression cases passed. An independent PyTorch complex
reference passed 120 combinations spanning 2D/3D inputs, widths 2/6/10/96/192,
float32/float16/bfloat16, explicit reordered positions, and forward/inverse
rotation. Five additional strided-view sentinel cases showed no overwrite.

The imported source was
`/job/repo/python/sglang/kernels/ops/attention/deepseek_v4_rope.py`. The change
is Python/Triton only; there is no native C++ library to rebuild, and the
prepared environment records `native: null`.

Validation used one AMD Instinct MI350X (`gfx950`) with Torch 2.11.0+rocm7.2,
ROCm 7.2.26015, and Triton 3.7.0. The exact contiguous width-6 example did not
show the NVIDIA report's race-dependent numerical mismatch on this AMD run,
but the base did independently and deterministically corrupt sentinel storage.
No NVIDIA B200 or `compute-sanitizer` was available, so the original NVIDIA
diagnostic was not reproduced. No full-model or distributed test was run.

Raw command output is retained in `raw/`.
