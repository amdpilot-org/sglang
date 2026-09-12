# Independent review of DSA EAGLE context-boundary candidate

Upstream issue: https://github.com/sgl-project/sglang/issues/30570

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2373

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2473

Candidate: https://github.com/amdpilot-org/sglang/pull/2441 at
`d7771df9f850e6826771f552b4690a7a6b639ee2`.

## Recommendation

**Accept as test-only hardening, not as a fully verified original-issue fix.**

The candidate is a single commit directly on the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. It changes only a registered GPU
test and investigation artifacts; production and native source are identical to
the base. The regression passes at the exact candidate and also passes when the
same test is run against the prepared base. That is expected because the base
already allocates the graph page table from `req_to_token.shape[1]`.

The test is useful: it would fail against the v0.5.14 allocation rule because a
614408-column request table would be copied into a 614406-column graph table,
and it covers smaller no-reserve, exact-legacy-width, and beyond-legacy-width
cases. However, it constructs the backend with `__new__` and manually performs
the target-verify-shaped copy. It does not call `_apply_cuda_graph_metadata`,
capture/replay an NVIDIA CUDA graph, start GLM-5.2-FP8, or execute TP=8 EAGLE.
It therefore validates the corrected allocation/width invariant rather than the
complete reported serving path.

## Independent evidence

- The v0.5.14 source retrieved from the upstream tag allocates
  `max_context_len + speculative_num_draft_tokens`, while target verify derives
  `max_seqlen_k` from sequence length plus draft tokens. Replaying the reported
  shapes on the assigned GPU produced the exact 614406-versus-614408
  `RuntimeError`.
- The recorded base and exact candidate have no production-source diff. The
  base allocator uses `max_ctx_len = self.req_to_token.shape[1]`, and graph
  replay obtains its copy width from the allocated page table.
- The exact candidate regression passed: 2 tests and 3 subtests.
- The same candidate test passed unchanged against the prepared base: 2 tests
  and 3 subtests. Thus the candidate does not supply a new runtime correction;
  it hardens coverage for an existing correction.
- Imports resolved to `/job/repo/python/sglang/...`; Torch resolved to the
  prepared `/opt/venv` installation. No native source changed, so no native
  rebuild was applicable.

## Architecture and environment limitations

The assigned device was one AMD Instinct MI350X (`gfx950`) with ROCm 7.2 and
Torch 2.11.0. The report requires eight Blackwell GPUs, NVIDIA CUDA graphs,
TP=8, and ZhipuAI/GLM-5.2-FP8 weights. Those were unavailable. The full model,
EAGLE scheduler, DSA fused metadata kernel on sm100, distributed worker failure,
SIGQUIT shutdown, and Kubernetes restart behavior remain unverified.

Raw review evidence was preserved outside the checkout under `/job`, including
candidate/base pytest logs and XML, import paths, GPU properties, historical
source excerpts, and the exact shape-failure traceback.
