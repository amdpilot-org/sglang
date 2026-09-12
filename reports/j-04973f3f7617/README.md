# Independent review of PR 1217

Reviewed `https://github.com/amdpilot-org/sglang/pull/1217` at exact commit
`74ec83a71f7217b9c4af342c749ca37e9c477a63` against:

- Upstream issue: https://github.com/sgl-project/sglang/issues/37342
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1252

## Recommendation

Request changes. The candidate is useful test-only hardening, but it does not
fully resolve or verify the original issue.

The recorded base reproduces the backend-selection test's ROCm host-platform
leak, and the candidate fixes that test isolation. The candidate also adds a
deterministic `_load_w13` copy/sharding test using the reported `[2048, 2048]`
weight and `[2048, 128]` scale shapes. Both tests pass at the exact candidate.

However, there is no production-code change. The loader test bypasses FusedMoE
and MXFP4 quant-method initialization, manually allocates destination tensors
from the checkpoint shapes, calls `_load_w13` directly, and never invokes the
FlashInfer SM90 post-load preprocessing or kernel. An independent adversarial
test showed the same fixture pattern accepts `[2048, 1024]` weights with
`[2048, 64]` scales and also accepts an invalid `[2048, 127]` scale width. It
therefore proves row sharding and W13 placement, not the hidden-size/scale
contract implicated by the original assertion.

## Evidence

- `evidence/base-selection.txt`: recorded base test fails because physical ROCm
  state leaks into a nominal SM100 case.
- `evidence/candidate-regression.txt`: exact candidate passes five tests and
  three subtests.
- `evidence/adversarial-loader.txt`: direct loader accepts dimensions outside
  the reported hidden-size/scale relationship.
- `evidence/candidate-sm90.txt`: Hopper suite skipped on this host.
- `evidence/import-paths-candidate.txt`: source was imported from this checkout;
  Torch is ROCm 7.2.
- `evidence/gpu.txt`: assigned AMD gfx950 GPU executed a numerical matmul. This
  does not qualify NVIDIA FlashInfer behavior.

## Limitations

DeepSeek-V4-Flash-0731 weights and four H800/SM90 GPUs were unavailable. Full
checkpoint loading, distributed TP=4 execution, Hopper preprocessing/kernel
execution, DSPARK, HTTP serving, and semantic accuracy remain unverified. The
candidate changes no native source, so no native rebuild was applicable.
