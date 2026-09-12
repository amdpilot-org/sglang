# Independent review of PR 1423

Candidate: https://github.com/amdpilot-org/sglang/pull/1423  
Exact candidate commit: `f6c401990d7325ba5bf2e9386560bc10805b7095`  
Parent candidate: https://github.com/amdpilot-org/sglang/pull/1247  
Parent independent review: https://github.com/amdpilot-org/sglang/pull/1337

Recommendation: **accept**. The candidate fully resolves the original reported
capacity/depth contract.

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a direct
production `NgramCorpus` at capacity 4 and maximum trie depth 4 constructed,
queued one four-token insertion, and aborted with `munmap_chunk(): invalid
pointer` (exit 134). This independently reproduces the issue's native failure.

At the exact candidate commit, the same direct native path uses a freshly built
JIT library and rejects 4/4 synchronously in its constructor with a descriptive
`RuntimeError`. The serving validation also rejects equality and below-depth
settings with `ValueError` under `python -O`; independent controls accept the
minimum valid 5/4 boundary and leave a non-NGRAM configuration unaffected.
The candidate's five serving-argument regressions and all 45 focused native
NGRAM corpus tests pass.

Import and build provenance are retained in
`raw/candidate_import_native_provenance.log`. Python loaded from `/job/repo`,
the build graph names the candidate checkout's five NGRAM C++ sources, and the
new diagnostic is embedded in the rebuilt shared library.

The prepared host has one AMD Instinct MI355X (`gfx950`) with ROCm 7.2, not the
reported NVIDIA L40S/CUDA environment, and the Qwen3-4B weights were not
available. No HTTP/full-model, NVIDIA-specific, semantic-accuracy, or
distributed-workload claim is made. The reproduced fault and corrected guards
are CPU-side, so GPU execution was neither necessary nor claimed.

The only non-product issue observed is trailing whitespace in archived report
artifacts already included by the candidate; product source and test code were
not implicated. No remaining counterexample to the original contract was
found.
