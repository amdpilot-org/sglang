# Independent review of PR 1332

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/1332 at exact commit `4715cafc4ac41e4fb29369ed4afef8d52c386399`

Upstream issue: https://github.com/sgl-project/sglang/issues/36830

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1364

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Recommendation: **unverified** for the full original issue. The candidate is a substantive source fix, not test-only hardening, and its zero-tail correction and raw-FP8 numerical path were independently verified on the assigned ROCm GPU. No source counterexample was found in the exercised boundaries. However, the original contract is specifically CUDA SM90/H20 with GLM-5.3-Flash, and that architecture and the model weights were unavailable. Therefore this review cannot honestly mark the original issue fully resolved by the exact candidate commit.

## Failing before

The prepared checkout was clean, on the requested branch, and exactly at the recorded base. Imports resolved to the checkout (`/job/repo/python/sglang/...`) under `/tmp/amdpilot-repo-j-192dd07a16a5/venv/bin/python`.

On the base, a direct CUDA-contract validation probe for `fp8_e4m3` with TileLang prefill and decode raised the existing `ValueError` saying TileLang FP8 KV is ROCm-only. The base regression suite passed 5 tests because those tests encode rejection of the requested configuration; this reproduces the original source-level blocker rather than proving functionality.

## Exact candidate review

The checkout was temporarily detached at exact commit `4715cafc4ac41e4fb29369ed4afef8d52c386399`, then returned to `amdpilot/j-192dd07a16a5` before this report was committed. Candidate imports continued to resolve to `/job/repo/python/sglang/...`; TileLang loaded from the prepared `/opt/tilelang/build` development installation.

The combined candidate:

- permits CUDA SM89+ only when both DSA consumers are TileLang;
- rejects mixed layouts and DCP greater than one;
- selects the raw 512-byte MLA layout and raw fused-quant writer;
- dispatches FP8 TileLang on CUDA and disables the scaled-layout MHA one-shot path;
- compile-time guards every zero-width tail allocation/copy/GEMM for GLM-5.3-Flash's `qk_rope_head_dim=0`.

No C++ or other native extension source changed, so a FlyDSL native rebuild was not applicable. The relevant implementation is Python/TileLang DSL and was compiled at runtime from the candidate checkout.

## Candidate regression and independent adversarial evidence

The candidate's focused suite completed with `11 passed, 2 skipped`. Its cross-vendor zero-tail regression compiled and executed on gfx950. The two CUDA-only numerical cases were skipped, as expected on ROCm.

An independent probe strengthened the one-hot regression:

- zero tail (`d_tail=0`), seven valid indices including a duplicate and scrambled order, with padding: compiled and executed; relative error against an independent FP32 softmax/gathered-KV reference was `0.027627935633063316`, below the candidate's 4% FP8 tolerance;
- nonzero tail (`d_tail=64`) one-hot boundary: compiled and executed; relative error was `0.0`.

Backend-resolution integration tests also passed: `1 passed, 93 deselected, 6 subtests passed`. `git diff --check` was clean.

## External evidence and limitations

Upstream PR https://github.com/sgl-project/sglang/pull/36904 contains an independent report of the same raw-layout approach working on 8x H20/SM90 with GLM-5.3-Flash and delivering 1.795x KV token capacity. It also contains the later report identifying the zero-tail compile failure and prescribing the four guards present in this candidate. This is strong corroboration, but it is not local execution of exact candidate commit `4715caf`.

The assigned device was AMD Instinct MI350X/gfx950 with ROCm 7.2 and Torch `2.11.0+rocm7.2`, not NVIDIA H20/SM90. GLM-5.3-Flash weights were unavailable. No full-model serving, CUDA graph capture, speculative decoding, semantic-accuracy suite, capacity measurement, multi-GPU, or multi-node run was performed. A reported raw-FP8 HumanEval quality decrease on sm_121 also remains outside this review's local verification.

Raw evidence is retained outside the revision-switching checkout at `/job/review-evidence-j-192dd07a16a5/`.
