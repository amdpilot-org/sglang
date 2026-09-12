# Independent review of candidate 87a8b5c3cc8f4de4c93c937c086612634786d60b

Upstream issue: https://github.com/sgl-project/sglang/issues/30122

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2392

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2450

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2415

## Recommendation

Accept the candidate as a narrow, source-level fix for reported bug 3 only. It does not fully resolve the original issue, which reports nine failures in a multi-node Qwen3 MoE GGUF deployment.

At the recorded base commit, `GGUFMoEMethod.apply()` unconditionally reads `self.fused_experts`, but neither `GGUFMoEMethod` nor `FusedMoEMethodBase` defines that attribute. An independent first-forward harness reproduced `AttributeError: 'GGUFMoEMethod' object has no attribute 'fused_experts'` before the mocked GGUF kernel was called. At the exact candidate commit, the candidate's two focused tests passed, and an independent harness confirmed that the same input reaches `fused_moe_gguf` and returns its result.

Removing the assertion matches the current implementation: `GGUFMoEMethod` directly invokes `fused_moe_gguf`; there is no alternate `self.fused_experts` implementation to select or validate. No native source changed, so a native rebuild was not applicable.

## Scope and counterexamples

This is a partial fix, not a full original-issue fix. Bugs 1, 2, and 4 through 9 were not changed by the candidate. In particular, an independent adversarial input using a non-standard/bypassed top-k representation still fails before kernel invocation because `apply()` assumes a three-item `StandardTopKOutput`. This is consistent with a concrete additional failure reported in the upstream discussion and is outside the candidate's stated bug-3 scope.

The candidate regression mocks `fused_moe_gguf`; it proves Python dispatch and argument forwarding, not numerical correctness of GGUF kernels or model inference. The review host has one AMD Instinct MI355X (`gfx950`) under ROCm 7.2, while this checkout warns that GGUF quantization is supported only on CUDA and MUSA. The reported two-GX10, TP=2, multi-node setup and Qwen3-235B-A22B GGUF weights were unavailable. Therefore no real GGUF GPU kernel, full model, tensor-parallel, or multi-node execution is claimed.

## Evidence

- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365` (the image-prepared checkout matched it exactly).
- Candidate: `87a8b5c3cc8f4de4c93c937c086612634786d60b`.
- Source import: `/job/repo/python/sglang/srt/layers/quantization/gguf.py`.
- Interpreter: `/tmp/amdpilot-repo-j-6fd17efaaf84/venv/bin/python`.
- Torch: `2.11.0+rocm7.2` from `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`.
- Candidate focused tests: 2 passed, 1 deselected.
- Candidate compile check: passed.
- Independent base harness: reproduced the unconditional `AttributeError`; kernel call count remained zero.
- Independent candidate harness: standard dispatch returned the mocked kernel output; a bypass-shaped top-k object failed with `TypeError`.
- `git diff --check` found trailing whitespace in the candidate's retained pytest transcript. This is report-artifact hygiene, not a functional defect in the fix.
