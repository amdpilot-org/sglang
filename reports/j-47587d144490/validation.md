# Correction validation

Candidate: https://github.com/amdpilot-org/sglang/pull/2914 at exact commit
`1baf5337c192902f8f19b56e05d713945b5ab4ff`

Independent review: https://github.com/amdpilot-org/sglang/pull/2997

The exact candidate was applied to the recorded base before any correction. The
independent native-boundary probe in `raw/native_overlap_probe.py` failed with
`max_active_native_loads=1` and roughly 0.40 seconds elapsed for two controlled
0.20-second native loads (`raw/native-overlap-before.txt`). Source inspection
confirmed that `_load_native_with_context` held `model_construction_lock` over
the complete `load_native` call, including Transformers/Diffusers
`from_pretrained` checkpoint I/O and weight materialization.

The correction removes only that outer whole-load lock. It retains the
candidate's parallel scheduler, deterministic assembly, opt-out, distributed
fallback, and the narrower lock in SGLang's known process-global construction
contexts. The identical probe then passed with `max_active_native_loads=2` and
roughly 0.20 seconds elapsed (`raw/native-overlap-after.txt`). A permanent unit
regression exercises `_load_native_with_context` directly.

The focused suite passed 32 tests, the extended compatibility suite passed 68
tests plus 9 subtests, and the one-GPU NumPy-reference regression passed on an
AMD Instinct MI350X. Raw output and import/environment paths are retained under
`raw/`.

This correction does not claim unavailable evidence. Qwen-Image weights were
not present, and real-model cold start, wake/refit, production pinned mappings,
multi-rank loading, and representative peak host memory remain unresolved as
detailed in `result.json`.
