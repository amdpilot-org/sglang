# Correction-generation investigation

This checkout preserves the valid test additions from
https://github.com/amdpilot-org/sglang/pull/2728 at exact commit
`8ae549b1d88e2d33a945914af80a20efea8e63e1` and independently investigates
the concrete counterexamples reported by
https://github.com/amdpilot-org/sglang/pull/2781.

Upstream issue: https://github.com/sgl-project/sglang/issues/34899

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2815

## Result

The candidate still collects 24 cases, including baseline, prefill-cache-hit,
and decode-cache-hit cases for both Python and Rust unified radix trees. The
candidate's Python unified-memory class was launched with the actual Inkling
configuration. It resolved unified memory, Triton attention, deterministic
inference, disabled prefill graphs, and the intended model revision, but did
not complete server startup or an HTTP request.

The assigned accelerator is an AMD Instinct MI350X (`gfx950`) under ROCm 7.2.
Importing `sglang.srt.models.inkling` fails because Inkling unconditionally
reaches its NVIDIA CUTE/FA4 implementation and the pinned environment has no
`cutlass` Python module. Consequently no Inkling KL value was measured, neither
tree backend completed an end-to-end request, and no real unified-memory
FULL/SWA/MAMBA restoration defect can be injected and observed by these new
classes on this host.

The existing unified MLA GPU parity test passed bit-for-bit against the stock
pool reference (3 tests and 4 subtests). This confirms that GPU execution is
available, but it does not cover Inkling, SWA, MAMBA checkpoints, radix cache
hits, or either candidate class. It is therefore retained only as scoped
component evidence.

No state-restoration source correction was made. Without a supported NVIDIA
CUTLASS/FA4 execution path and Inkling weights, such a change would be
speculative. In particular, this environment cannot provide the requested
revert-then-red/failing-before and passing-after numerical evidence. The tiny
Llama fixture is not a substitute because it has none of Inkling's combined
FULL+SWA+MAMBA architecture.

## Reproduction

```bash
/tmp/amdpilot-repo-j-d13107cd7c7a/venv/bin/python -m pytest --collect-only -q \
  test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_hybrid_bitexact.py

/tmp/amdpilot-repo-j-d13107cd7c7a/venv/bin/python -m pytest \
  test/registered/radix_cache/unified_radix_tree/test_unified_radix_cache_kl_hybrid_bitexact.py::TestUnifiedMemoryHybridBitExact::test_logprobs_match \
  -vv -s

/tmp/amdpilot-repo-j-d13107cd7c7a/venv/bin/python -m pytest \
  test/registered/unit/mem_cache/test_unified_mla_gpu_parity.py -vv -s
```

Raw evidence is in this directory. `candidate-python-e2e.log` records the
actual launch up to the unsupported model import/startup boundary;
`inkling-import.log` isolates the complete `ModuleNotFoundError` traceback.
