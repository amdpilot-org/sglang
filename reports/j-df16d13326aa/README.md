# Investigation of SGLang issue 37478

Upstream issue: https://github.com/sgl-project/sglang/issues/37478

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3329

The underlying aiter compiler failure reproduces on the assigned gfx950 GPU at
the reported 2 GiB boundary. `23168 x 23168` fp32 logits (2,147,024,896 bytes)
return successfully, while `23180 x 23180` (2,149,249,600 bytes) abort the
subprocess in LLVM with exit code 134.

The reported SGLang call path is not present at the recorded base. The current
`IndexerKPool` has no `_fp8_mqa_logits` method, imports no aiter logits kernel,
and its `_should_chunk_mqa_logits` helper still has zero callers. Its two
non-paged logits sites call `deep_gemm.fp8_mqa_logits` directly. On this ROCm
environment `sglang.srt.utils.is_cuda()` is false and `deep_gemm` is not
installed, so this checkout does not provide the reported KPool aiter path to
patch or execute.

No product-code change was made. Re-adding the reverted aiter HIP path merely
to apply the proposed bound would be a broader re-land of unrelated GLM/KPool
support and would not be justified by the checked-out implementation. The
retained scripts and logs independently establish both the dependency boundary
and the source-path blocker.

## Evidence

- `raw/aiter-23168.log` and `.exit_code`: below-boundary GPU kernel return.
- `raw/aiter-23180.log` and `.exit_code`: above-boundary LLVM assertion abort.
- `raw/source-reachability.log`: AST/text inventory of the checked-out KPool.
- `reproduce_aiter_boundary.py`: isolated subprocess reproducer.
- `check_source_reachability.py`: deterministic checked-source inventory.

## Limitations

GLM-5.3-Flash weights were not available, so no full-model or serving-path
claim is made. The tiny Llama fixture is not applicable because it cannot
exercise the GLM KPool architecture. No multi-rank or multi-node test was run.
