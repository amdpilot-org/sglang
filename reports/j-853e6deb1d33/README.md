# Independent review of PR 1402

Reviewed exact candidate commit `f36cc319163a0d3c5be588d5e9165ac2fd91062e`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and upstream issue
https://github.com/sgl-project/sglang/issues/35498.

Recommendation: **accept** as test-only hardening. The candidate changes no
runtime source. The recorded base already implements the performance-preserving
solution: `Engine._serialize_tensors_per_rank` creates one independent
`MultiprocessingSerializer` payload per TP rank, and
`TpModelWorker._deserialize_own_rank` selects only the payload for its rank.

The issue-specific legacy topology was independently reproduced with eight
spawned consumers: one succeeded and seven raised `EOFError`, with seven
`resource_sharer` `KeyError` tracebacks. The base/candidate per-rank path then
passed three consecutive eight-consumer rounds and a TP1 boundary, including
independent checks of element count and tensor sum. The candidate's registered
test also passed all three cases.

This fully resolves the isolated one-shot-FD contract exercised by the original
CPU reproduction. It does not constitute an end-to-end TP8 Qwen3 server test:
the environment has one AMD Instinct MI350X (gfx950) GPU and no required model
weights. No GPU execution was necessary for this CPU FD-ownership defect. The
candidate contains no native changes, so no native rebuild was applicable.

Evidence is retained under `evidence/`. Runtime imports resolved to the prepared
checkout for SGLang source and to the pinned ROCm PyTorch installation.
