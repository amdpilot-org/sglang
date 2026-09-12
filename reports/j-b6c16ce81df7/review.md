# Independent review: PR 3142 at e0642eaba6485b97fbd8f0003f30adb08f442146

Recommendation: **accept**, with `fully_resolves_original=false` because NVIDIA/DeepGEMM and deployment-level claims remain architecture-blocked rather than proven.

The exact candidate fixes the concrete remaining counterexample from the prior independent review. At predecessor commit `4c7768d36ecfa2fb078ad51cfed49932677dfda0`, copying a completed cache beneath a different `DG_JIT_CACHE_DIR` changed the marker key and invoked exhaustive warmup twice. At reviewed commit `e0642eaba6485b97fbd8f0003f30adb08f442146`, the cache root is excluded from the payload, the copied marker name remains stable, and warmup is invoked once. Other `DG_JIT_*` values still invalidate compatibility.

This is a substantive source correction, not test-only hardening. The final commit adds retained evidence, but its parent `364dc0fbc8b4eb232368aab5a2118dfa5ccf8859` contains the behavior change. Source inspection also confirms that only `_maybe_compile_deep_gemm_one_type_all` can return early; CUDA graph capture, graph instantiation, graph-shape selection, and eager validation code are not modified by this candidate.

The prepared base independently reproduced redundant dispatch after cache transfer. The candidate's 13 regressions and three separately authored adversarial cases passed. Raw logs and test sources are preserved under `/job/review-evidence`, outside the checkout used for revision switching.

Architecture limitation: the assigned accelerator is an AMD Instinct MI355X under ROCm 7.2. `torch.version.cuda` is `None`, and `deep_gemm` is not installed. Consequently, no real NVIDIA DeepGEMM compilation/cache load, independent GPU numerical comparison, CUDA graph parity run, or DeepSeek-V4-Pro TP=8 benchmark was possible. No native source changed, so rebuilding native code was not applicable.

No additional concrete source-level counterexample was found. The recommendation accepts the bounded relocation correction and marker protocol while explicitly withholding a claim of complete deployment validation.
