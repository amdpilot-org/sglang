# Independent review of amdpilot-org/sglang PR 2889

Candidate reviewed exactly at `0d598f3188d250d251ffde2782d53e247c493aa9` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Recommendation

Request changes. The candidate is a substantial partial fix: it introduces the requested root and fixed layout, preserves documented override precedence, stops the SGLang Inductor adapter from moving already-configured caches, and works in a real ROCm Inductor/Triton compilation. It does not fully satisfy the DeepGEMM portion of the original contract.

## Failing before, passing after

On the base, `SGLANG_JIT_CACHE_ROOT` is absent/ignored. Importing SGLang initially points Triton and Inductor at `/job/.cache/sglang`, and `InductorAdaptor.initialize_cache()` then moves both into graph-local `inductor_cache/` and `triton_cache/` directories. The independent contract check exits 1.

At the candidate commit, the same check resolves `{root}/triton` and `{root}/inductor` during import and retains both locations after `InductorAdaptor.initialize_cache()`. It exits 0. Imports were confirmed from `/job/repo/python/sglang`, not from an unrelated installed copy.

## Remaining counterexample

The candidate says it configures DeepGEMM before importing it, but the executable import order does not do that on supported CUDA/MUSA hardware:

1. `compile_utils.py:19` imports `ENABLE_JIT_DEEPGEMM` from `configurer.py`.
2. Importing `configurer.py` immediately evaluates `_compute_enable_deep_gemm()` at line 39.
3. On a supported CUDA/MUSA architecture, `_compute_enable_deep_gemm()` imports `deep_gemm` at line 32 (and SM120 can import it even earlier at line 25).
4. Control only then returns to `compile_utils.py`, which assigns `DG_JIT_CACHE_DIR` at line 34.

Thus the first DeepGEMM import can observe its old/default cache location before the unified root is installed. The added regression test only compares the assignment with the later conditional import in `compile_utils.py`; it does not account for the earlier import through `configurer.py`.

This path could not be executed on the assigned AMD GPU because DeepGEMM is NVIDIA/MUSA-specific, but the Python import order is deterministic and directly contradicts the candidate's import-before-configuration claim. NVIDIA/MUSA kernel cache writes remain unverified.

## Environment and scope

- Prepared interpreter: `/tmp/amdpilot-repo-j-9f1fdc289b80/venv/bin/python`
- Torch: `2.11.0+rocm7.2`; HIP: `7.2.26015`
- GPU: one visible AMD Instinct MI350X
- Candidate source imports: `/job/repo/python/sglang`
- Native code changed: no; native rebuild: not applicable
- Verified on ROCm: Triton and Inductor path placement plus generated `.amdgcn`/`.hsaco` artifacts and numerical output
- Not executable here: DeepGEMM, CUDA driver cache, FlashInfer CUDA cache, and MUSA behavior
- No serving fixture was used because the affected contract is compiler-cache configuration and was exercised directly through real GPU compilation.

## Classification

This is a partial original-issue fix, not test-only hardening and not a full fix. The main Triton/Inductor behavior is verified; the DeepGEMM import-order component remains defective, and NVIDIA/MUSA-specific writes are architecture-blocked.
