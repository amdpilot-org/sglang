# Independent review of PR 3359

Upstream issue: https://github.com/sgl-project/sglang/issues/37936

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3323

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3360

Reviewed exact candidate commit: `2c2f2113585144c51d1f0d1a1aa816b7f552c1dd`

## Conclusion

The candidate is valid **test-only hardening**, but it does not independently
establish a full fix for the original serving failure. The recorded base
already calls `EagleDraftExtendInputBuffers.share_buffers` with
`exclude={"select_index"}` at the production EAGLE draft-extend construction
site. The candidate moves that existing rule into an override on the buffer
class and adds a direct-buffer regression.

The submitted failing-before result is real for its direct-buffer fixture, but
that fixture deliberately calls `share_buffers()` without the exclusion that
the recorded base's production caller already supplies. It therefore proves
the value of making the invariant intrinsic and protects hypothetical future
or alternate callers; it is not reproduction of the original base serving
failure. Repository search found no second production constructor/caller that
was missing the base exclusion.

The candidate regression and the adjacent EAGLE selected-logits suite pass,
and an independent adversarial case confirms that a caller-supplied exclusion
is unioned with the mandatory `select_index` exclusion. This supports accepting
the hardening on its stated narrow merits. It does not prove that cross-width
buffer aliasing is the sole cause of the reported concurrent constrained-tool
crash, nor that the original workload is fixed.

## Commands and evidence

- On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`:
  `/tmp/amdpilot-repo-j-acd52e73b004/venv/bin/python -m pytest -q test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py test/registered/unit/spec/test_eagle_draft_extend_logits.py`
  passed 46 tests. This also confirms the prepared checkout matched the
  recorded base.
- On exact candidate `2c2f2113585144c51d1f0d1a1aa816b7f552c1dd`, the same
  command passed 46 tests. The candidate replaces a generic exclusion test
  with the EAGLE-specific regression, so the aggregate count is unchanged.
- An independent direct construction called
  `share_buffers(exclude={"input_ids"})` on two instances and verified both
  `input_ids` and `select_index` remained distinct. This exercises the
  candidate's union semantics rather than repeating its regression.
- A safe GPU gather on the assigned AMD Instinct MI350X, gfx950, returned
  `[11, 13]` for indices `[1, 3]` over `[10, 11, 12, 13]`.
- `git diff --check` passed for the exact base-to-candidate diff.

The interpreter imported SGLang source from the prepared checkout and Torch
`2.11.0+rocm7.2`. No native C++/CUDA/HIP sources changed, so no native rebuild
was applicable.

## Limitations and remaining counterexamples

The environment supplied one AMD gfx950 GPU, not 4 Tesla V100 GPUs. The
Qwen3.8 Flash-Next NVFP4 weights, V100-specific CUDA image/backends, TP=4
PCIe-only topology, and the complete concurrent HTTP workload were unavailable.
Accordingly the CUDA device assertion, NCCL abort, Xid 43, structural-tag
failure interaction, chunked insertion during active EAGLE decode, and engine
survival after that exact workload remain untested. A second stale-index or
scheduler-state path unrelated to cross-width buffer pooling remains a valid
counterexample to a full-resolution claim.
