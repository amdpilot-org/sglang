# Independent review of PR 758 at 811ed2e

Upstream issue: https://github.com/sgl-project/sglang/issues/38795

Mirror issue: https://github.com/amdpilot-org/sglang/issues/783

Candidate: https://github.com/amdpilot-org/sglang/pull/758 at `811ed2e397dc535fc7ec569c2e6f9a6be4b4181d`

Recommendation: accept.

The prepared base reproduced the issue through the actual `ModelOptNvFp4FusedMoEMethod.apply` method. With a cached per-layer `FLASHINFER_CUTLASS` backend and a live global `TRITON` or `AUTO`, both cases raised the contradictory error naming cached `FLASHINFER_CUTLASS`. The exact candidate changed the apply branch guards to use the same `moe_runner_backend` local as the diagnostic, and the failing cases passed.

The candidate's regression passed 8/8. An independent suite passed 5/5 and checked both directions of adaptive-like state changes, missing cache fallback, routed TRT-LLM, and source-level coverage of every supported local backend predicate. External CUDA implementations were stubbed only where `apply` hands quantization metadata to the runner.

The change is Python-only; no native source or library changed, so a native rebuild was not applicable. Imports were confirmed from `/job/repo/python/sglang/srt/layers/quantization/modelopt_quant.py` under the prescribed interpreter.

Architecture limitation: this host exposes one AMD Instinct MI355X with Torch 2.11.0+rocm7.2, not NVIDIA B300/CUDA/FlashInfer. A real GPU numerical probe ran successfully on the MI355X, but it does not qualify NVFP4 FlashInfer output. The original 8xB300 TP8/EP8 adaptive EAGLE CUDA-graph startup remains explicitly unverified.

One non-blocking residue remains: Python eagerly evaluates the default argument in `getattr(self, "_moe_runner_backend", get_moe_runner_backend())`, so the global getter is still called even with cached state. The candidate makes that returned global value non-authoritative for branch selection and diagnostics; no behavioral counterexample to the original contract was found.

Raw review evidence is retained outside the revision-switching checkout at `/tmp/amdpilot-repo-j-751324e2e79b/review-evidence/`.
