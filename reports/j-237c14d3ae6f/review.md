# Independent review of PR 1380

Upstream issue: https://github.com/sgl-project/sglang/issues/36481

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1418

Candidate: https://github.com/amdpilot-org/sglang/pull/1380 at exact commit `62688251058491fc6bb5251a95bc187c3019dea8`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate is a valid, narrow crash-avoidance fallback, but
it does not fully resolve the original contract: compact target verification on
a Triton-backed Qwen hybrid model still does not use a CUDA graph. It is routed
to eager execution because `TritonAttnBackend.supports_ragged_verify_graph` is
false and `HybridLinearAttnBackend` requires both child backends to opt in.

The candidate therefore prevents the reported boot-time capture crash without
making `extend_attention_fwd` safe under compact graph capture. This is a
partial fix, not a full original-issue fix. Its regression is useful control-flow
hardening, but remains mock-only.

## Reproduction and independent checks

The candidate regression was copied outside the checkout before revision
switching and run against both revisions with the prepared interpreter.

- On the recorded base, the candidate suite failed exactly at the new contract:
  unsupported compact capture entered the poisoned `warmup()`; five other tests
  passed. See `evidence/base-regression.log`.
- At the exact candidate commit, all six tests passed. See
  `evidence/candidate-regression.log`.
- Independent adversarial combinations confirmed that hybrid support is false
  unless both children opt in; unsupported compact capture returns without
  warmup; supported compact and unsupported non-compact capture continue. See
  `evidence/adversarial.log`.
- A real GPU tensor calculation matched an independent CPU reference. This only
  establishes that the assigned AMD device executes work; it is not evidence
  for the NVIDIA kernel bug.

The imported `sglang`, decode runner, Triton backend, and hybrid backend all
resolved from `/job/repo/python/sglang/...`, so tests exercised the checked-out
source rather than a separately installed package. No native/C++ source changes
exist in the candidate, so a native rebuild was not applicable.

## Architecture limitation

The assigned device is one AMD Instinct MI355X (`gfx950`) with Torch
`2.11.0+rocm7.2`, HIP `7.2.26015`, and no CUDA runtime. The report requires an
NVIDIA B300 with CUDA 13.2 plus unavailable Qwen3.5/Qwen3.6 hybrid and DSpark
weights. Consequently, the batch-size-20/eight-token illegal memory access was
not reproduced, and neither `extend_attention_fwd` nor a full-attention plus
GatedDeltaNet model was executed.

## Classification

- Full original-issue fix: no.
- Partial fix: yes; the crash-prone unsupported capture is skipped safely.
- Test-only hardening: the added regression validates capture/replay admission
  control with mocks but not the failing kernel or model architecture.
- Unverified claim: any claim that compact Triton-backed hybrid CUDA-graph
  execution is fixed on B300/CUDA 13.2 remains unverified.
