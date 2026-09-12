# Independent review of PR 2401

Upstream issue: https://github.com/sgl-project/sglang/issues/31719

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2436

Candidate: https://github.com/amdpilot-org/sglang/pull/2401 at exact commit
`8926f3ebb5219cba35b8fcd68916bde4bb8fe74c`.

## Recommendation

Request changes. The candidate is a partial fix: its ordinary GDN prefill path
handles mixed activation/cache dtypes and passes its numerical gfx950 fixture,
but the same original mixed-dtype compiler failure remains reachable through
`GDNAttnBackend._forward_mis_segments`. In addition, the new mismatch helper
does not preserve the causal-convolution padding contract.

## Evidence

The prepared checkout was exactly the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. On that revision, the actual GDN
indexed assignment was reproduced on CPU and GPU for FP16 activation/BF16
cache, BF16 activation/FP32 cache, and BF16 activation/FP16 cache. PyTorch
raised `Index put requires the source and destination dtypes match` in every
case (`raw/base_original_failure.txt`).

I then detached the checkout at the exact candidate commit and confirmed that
the imported source was `/job/repo/python/sglang/...`, not an installed copy.
The candidate's regression passed all four advertised dtype combinations on
the assigned AMD Instinct MI355X/gfx950, checking output and updated cache
against a grouped `torch.nn.functional.conv1d` reference
(`raw/candidate_regression.txt`). Its focused unit suite also passed 18 tests
and 27 subtests (`raw/candidate_unit.txt`).

Two independent adversarial checks failed the candidate contract:

1. `_forward_mis_segments` still calls `causal_conv1d_fn` directly rather than
   `_causal_conv1d_with_cache_dtype`. At the exact candidate commit, BF16
   activations with both FP32 and FP16 caches still fail Triton compilation
   with `Mismatched type for new_conv_state ...` (`raw/adversarial_mis_boundary.txt`).
   This is the same causal-convolution boundary and dtype pairs identified by
   the preceding independent review, now on a remaining production GDN path.
2. The causal-convolution API defines `PAD_SLOT_ID = -1` and the hybrid backend
   deliberately emits `-1` for padded rows. The direct same-dtype kernel left
   that row unchanged. The candidate mismatch helper gathered
   `conv_states[-1]`, replaced indices with `[0, 1, ...]`, processed the padded
   sequence as real, and scattered the result back to row `-1`; the measured
   maximum unintended change was 4.28125 (`raw/adversarial_pad_slot.txt`).

The checkout was returned to `amdpilot/j-9a88264f6788` before this report was
created. The candidate was not modified or merged.

## Environment and limitations

GPU execution used one AMD Instinct MI355X (`gfx950`), Torch 2.11.0+rocm7.2,
and HIP 7.2.26015. The reported QuantTrio/Qwen3.6-27B-AWQ weights, NVIDIA RTX
5090, CUDA 13.3, and SM120 were unavailable, so no full-model serving,
NVIDIA-specific, semantic-accuracy, or multi-node claim is made. The tiny
Llama transport fixture cannot qualify a qwen3_5 hybrid-GDN path and was not
substituted. No native source changed in the candidate, so no native rebuild
was applicable; Triton kernels were freshly compiled from the candidate source
using a private cache.
