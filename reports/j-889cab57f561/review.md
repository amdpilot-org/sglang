# Independent review of candidate PR 2280

Upstream issue: https://github.com/sgl-project/sglang/issues/31719

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2215

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2316

Candidate: https://github.com/amdpilot-org/sglang/pull/2280 at exact commit `4b0d2276a4cf6ce3e6ed5d2c7f66dce81416d0ac`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`. The image-prepared checkout was already at that exact commit, so there was no base discrepancy.

## Recommendation

Request changes. The candidate is a useful partial fix for the reporter's primary FP16-activation/BF16-cache crash, but it does not fully resolve the original issue's stated cross-dtype contract. In particular, the original report explicitly includes BF16 activations with `SGLANG_MAMBA_CONV_DTYPE=float32`. The candidate casts the tracked snapshot before indexed assignment, but the same mixed-dtype cache is immediately passed to the production causal-convolution kernel. On the assigned ROCm gfx950 GPU, that subsequent call fails during Triton compilation because its branches produce FP32 and BF16 state values.

The candidate tests only the extracted assignment helper. They correctly prove that the `index_put` operation no longer rejects mismatched dtypes, but they do not execute the following production convolution call and therefore miss the remaining failure.

## Evidence

On the prepared base, direct indexed writes reproduced the reported `Index put requires the source and destination dtypes match` exception on both CPU and gfx950 for BF16-cache/FP16-source, FP16-cache/BF16-source, and FP32-cache/BF16-source pairs. See `raw/index_put_repro.txt`.

At the exact candidate commit, the candidate regression suite passed: 17 tests, 27 subtests. Independent GPU cases also confirmed that `_store_tracked_conv_states` converts non-contiguous sources into the cache dtype and preserves unselected slots for four dtype combinations. See `raw/pytest_candidate.txt` and `raw/adversarial_gpu.txt`.

An independent production-path boundary test then called `sglang.srt.layers.attention.mamba.causal_conv1d.causal_conv1d_fn`, the function invoked immediately after the changed assignment in `GDNAttnBackend.forward_extend`. FP16 activations with a BF16 cache succeeded on gfx950. BF16 activations with an FP32 cache failed with a Triton `CompilationError`: `Mismatched type for new_conv_state ... fp32 ... bf16`. BF16 activations with an FP16 cache failed analogously. See `raw/production_conv_mixed_dtype.txt`.

Thus the candidate fixes the exact first failing operation in the primary reported configuration but does not establish that intentionally different cache and activation dtypes are supported end-to-end. A regression should exercise the assignment plus causal-convolution boundary, especially the explicit FP32 override from the issue, or the configuration policy should prevent unsupported mixed dtype combinations.

## Source and environment

The prepared interpreter imported SGLang from `/job/repo/python/sglang` and `gdn_backend.py` from that checkout, not from an installed SGLang wheel. Torch was `/opt/venv/lib/python3.12/site-packages/torch`, version `2.11.0+rocm7.2`, HIP `7.2.26015`. The assigned device was one AMD Instinct MI355X, gfx950. The candidate changes Python and tests only; it changes no native source, so no native rebuild was applicable.

The reported QuantTrio/Qwen3.6-27B-AWQ weights, NVIDIA RTX 5090, CUDA 13.3, and SM120 were unavailable. No full-model serving, semantic-accuracy, NVIDIA, or multi-node claim is made. The GPU evidence qualifies the implicated state write and the immediately following ROCm Triton causal-convolution boundary only.
