# Independent review of PR 526

Upstream issue: https://github.com/sgl-project/sglang/issues/38319

Mirror issue: https://github.com/amdpilot-org/sglang/issues/630

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/526 at
`4d68007d1345b03fd894dcc23e56f3f2bc687f8f`.

Recommendation: **reject**. The candidate changes only a test and reports; it
does not change the implementation. Its new test manually calls
`allocator.free(request_table_view)` and then mutates that view. Neither real
paged cache path tested here retains that view:

- Python `RadixCache.cache_unfinished_req` converts the sliced representatives
  to page IDs before deferring them.
- `RadixCacheCpp.cache_unfinished_req` copies the request-table indices with
  `to(dtype=torch.int64, copy=True)` before deferring a free.

Both real paths retained correct page ownership even after the candidate's
claimed fix (`_copy_for_free_group`) was deliberately disabled. Thus the new
test demonstrates that cloning protects a synthetic direct-view sequence, but
does not establish that the sequence is reachable from chunked prefill,
radix insertion, retraction, or abort.

The exact candidate regressions pass, but that is test-only hardening for an
unproven sequence. The original persistent QSA corruption was not reproduced
on the prepared base and is not fully resolved by this candidate.

## Environment and limitations

The prepared machine has one AMD Instinct MI355X (`gfx950`) under ROCm 7.2 and
Torch 2.11.0+rocm7.2. It is not the reported DGX Spark GB10 / SM121 CUDA
architecture. The Qwen3.8-Flash-Next-NVFP4 weights, QSA KDA serving setup,
NVFP4 KV cache, EAGLE configuration, long-context memory pressure, abort timing,
and impossible token 248319 were unavailable. End-to-end model logits and the
reported stochastic serving failure therefore remain unverified.

Python imports resolved to `/job/repo/python/sglang`. The C++ radix wrapper
resolved to the same checkout and loaded `radix_tree_cpp` through its
checkout-local `torch.utils.cpp_extension.load` source definition. The
candidate changes no native source, so a native rebuild was not applicable.
The candidate's committed GPU log names an MI350X, while this review's direct
device query reports MI355X; its archived environment claim was not accepted as
independent proof.

