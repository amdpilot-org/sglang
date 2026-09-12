# Independent review of PR 590 at 45638ad

Recommendation: **request changes**. The candidate substantially implements the new `skip_cache_insert` feature, but it does not fully resolve the original issue because it intentionally removes backward compatibility for the existing `bootstrap_host: "2.2.2.2"` sentinel.

The prepared base has no public field: `GenerateReqInput(..., skip_cache_insert=True)` raises `TypeError`, while the OpenAI request models discard that unknown input. At the exact candidate commit, the field is present and correctly propagated through native generation, OpenAI chat/completions/responses, batch normalization, tokenizer/scheduler request construction, sessions, and cache release paths.

The candidate's new behavior passed 241 focused unit tests plus 81 subtests. It also passed five live GPU tests against both plain radix cache and HiCache `write_through` on an AMD Instinct MI350X. These checks covered short and multi-chunk prompts, no-insert follow-ups, reads from an existing prefix, mixed per-item batch flags, and OpenAI chat.

## Blocking finding

The original issue requires: `Req.__init__` should use `skip_cache_insert or bootstrap_host == FAKE_BOOTSTRAP_HOST`, and “The PD sentinel keeps working unchanged.” The base does exactly that sentinel derivation. The candidate instead assigns only `self.skip_radix_cache_insert = skip_cache_insert`, removes the sentinel import, and adds a regression test asserting that the fake bootstrap host alone does *not* skip.

That breaks existing external clients using the documented-in-issue workaround. A request with only `bootstrap_host: "2.2.2.2"` changes from no insertion on the base to cache insertion on the candidate. Updating known internal synthetic callers does not preserve behavior for external callers. The fix should retain the logical OR with the sentinel while also plumbing the explicit field.

## Environment and scope

- Python imports resolved to the candidate checkout under `/job/repo/python/sglang`.
- PyTorch was `2.11.0+rocm7.2`, HIP `7.2.26015`; GPU was MI350X/gfx950.
- No C++ or FlyDSL source changed, so a native rebuild was not applicable. `radix_cache_cpp.py` is Python and its focused tests passed.
- The live run used `Qwen/Qwen2.5-1.5B-Instruct` because the repository default is a gated Llama model.
- Prefill-only PD, speculative decoding, retraction, and live pure-SWA paths remain unverified end to end.
- During live startup Aiter detected that pinned clang rejected `-amdgpu-coerce-illegal-types=1`, retried without the flag, built gfx950 kernels, and completed all inference assertions.

Raw evidence is retained outside the checkout at `/job/review-evidence-j-dff2bedd469e/`.
