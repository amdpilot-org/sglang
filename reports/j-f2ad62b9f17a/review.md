# Independent review of PR 662

Upstream issue: https://github.com/sgl-project/sglang/issues/22935

Mirror issue: https://github.com/amdpilot-org/sglang/issues/667

Candidate: https://github.com/amdpilot-org/sglang/pull/662 at
`6eaf6c114f3dd23ad901e47d2cf8e2532ade22a6`.

## Recommendation

Reject as a fix for the original issue. The candidate is accurate test-only
hardening and review documentation, but it makes no runtime change and its new
test explicitly asserts that the original exact `[A,B,C]` then n-1 `[A,B]`
lookup returns zero indices. It therefore cannot be a failing-before,
passing-after regression for the reported contract.

## Independent result

The prepared base and exact candidate behaved identically on an assigned AMD
Instinct MI350X with Torch `2.11.0+rocm7.2`:

- a depth-3 leaf split by matching `[A,B]` returned 0 indices;
- matching the full owned leaf returned 3 indices;
- separately inserting a real state owned at depth 2 returned 2 indices;
- a real depth-64 checkpoint returned 64 indices for a 95-token n-1 lookup;
- divergence after depth 64 returned 64 indices;
- divergence before the first checkpoint returned 0 indices.

The nonzero device results were compared with independently constructed GPU
`torch.arange` references. Source imports resolved to
`/job/repo/python/sglang`, including
`python/sglang/srt/mem_cache/mamba_radix_cache.py`.

## Architectural constraint

The state stored after `[A,B,C]` belongs to depth 3 and cannot be assigned to
the split depth-2 node without representing the wrong recurrent and
convolution state. The scheduler treats returned prefix indices as already
computed. The reviewed code has neither an arbitrary-depth depth-2 snapshot
nor a path that keeps attention KV while replaying the same prefix through the
recurrent layers. Exact nonzero reuse consequently needs one of those
architectural capabilities.

PR 662 correctly documents this ownership constraint and protects supported
long-prefix behavior, but that is not a full or partial runtime fix for the
original short-prefix case. It is test-only hardening.

## Environment and rebuild scope

The candidate changes reports and one Python unit test only. It changes no
runtime Python implementation, C++, HIP, CUDA, Rust, FlyDSL, or other native
source, so no native rebuild was applicable. No model weights were prepared;
end-to-end hybrid-model logits and performance remain unverified.
