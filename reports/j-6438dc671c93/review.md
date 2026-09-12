# Independent review of PR 934

Upstream issue: https://github.com/sgl-project/sglang/issues/22935

Mirror issue: https://github.com/amdpilot-org/sglang/issues/967

Candidate: https://github.com/amdpilot-org/sglang/pull/934 at
`1aa0bfa420a40d0da35caf7710f14e27ebf8c479`.

## Recommendation

Reject as a solution to the original issue. The candidate is test/report-only
hardening: its parent is the recorded base, and its diff contains no runtime or
native source changes. Its new n-1 test passes by asserting that the reported
zero-hit remains present.

## Independent reproduction

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, an
actual `MambaRadixCache` GPU probe inserted `[10,20,30]` with the only Mamba
state at depth 3, then matched `[10,20]`. It returned zero device indices, made
the split parent state-less, and preserved the state on the depth-3 leaf. This
reproduces the original issue rather than a neighboring smoke.

After temporarily checking out the exact candidate, its focused suite passed 20
tests and its GPU probe produced the same zero-hit. Positive controls showed
that exact owned checkpoints at depths 2 and 37 return the expected index
vectors, while divergence at depth 31 before the first depth-64 checkpoint
returns zero. Thus the behavior is checkpoint-ownership-dependent, not a general
failure of radix matching.

## Classification

- Full original-issue fix: no.
- Partial runtime fix: no; runtime source is byte-for-byte unchanged from base.
- Test-only hardening: yes; the tests usefully encode that a descendant recurrent
  and convolution state cannot be relabeled as an earlier split checkpoint.
- Unverified claim: architecture-level alternatives remain unverified because no
  qualified hybrid-Mamba weights were available.

The concrete remaining counterexamples are unchanged: an n-1 split before the
only checkpoint yields zero indices; divergence before the first checkpoint
yields zero despite a structural attention-KV match; and there is no execution
path that retains deeper attention KV while replaying recurrent layers, nor
production of complete recurrent-plus-convolution checkpoints at the desired
arbitrary depth.

## Environment and native-code audit

Tests ran with the required interpreter, Torch `2.11.0+rocm7.2`, and an AMD
Instinct MI355X. `sglang` and `mamba_radix_cache` imported from the checked-out
`/job/repo/python` tree. The candidate changes no C, C++, HIP, CUDA, Cython,
CMake, or runtime Python source, so a native rebuild was neither required nor
applicable.

No hybrid-Mamba weights were prepared. Consequently, cold/warm serving logits
and logprobs and proposed architecture-specific replay/checkpoint paths were not
validated. The qualified tiny Llama fixture cannot establish Mamba correctness.
This limitation does not affect the direct cache-level reproduction.
