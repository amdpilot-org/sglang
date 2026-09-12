# Unified-memory bit-exact coverage report

The source request's remaining roadmap item is exact-KL coverage for unified
memory. Existing coverage split the relevant properties across two suites:

- `test_unified_radix_cache_kl_hybrid_bitexact.py` enforced the exact oracle on
  Inkling's FULL + SWA + MAMBA model, but did not enable unified memory.
- `test_inkling_unified.py` enabled unified memory, but accepted KL below
  `1e-2` and repeated-prefix logprobs equal only to three decimal places.

This change adds dedicated unified-memory exact-KL classes for both Python and
Rust tree cores. Each class pins deterministic inference and the unified-memory
compatible Triton/page-major path, verifies the resolved runtime configuration,
and runs the prefill/decode baseline plus explicit prefill- and decode-cache-hit
restoration checks at the existing `1e-9` exactness guard.

## Validation and blocker

The assigned GPU was an AMD Instinct MI355X (`gfx950`). The pre-change Inkling
serving test was attempted first, but server startup failed before loading model
weights because `sglang.srt.models.inkling` imports the NVIDIA FA4/CUTLASS
implementation and the pinned ROCm environment has no `cutlass` Python module.
The subsequent model-registry error is a consequence of that failed import.
See `before-unified-logprob.log`.

The new module collects all six added cases successfully. On the available GPU,
the independent-reference unified MLA pool test passed bit-for-bit for random
locations, two page sizes, and both Triton and JIT paths. Unified MAMBA and SWA
view/translation unit tests also passed. Raw logs and JUnit files are retained in
this directory.

These lower-level results do not qualify the full feature. A supported NVIDIA
host with CUTLASS/FA4 must execute the new Inkling classes to establish exact
zero KL across the combined FULL, SWA, and MAMBA serving path. The tiny Llama
fixture is intentionally not used because it cannot exercise that architecture.
