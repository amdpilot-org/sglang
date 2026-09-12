# Correction generation 1: serving metric scope

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2577 at exact
commit `3516408f3178b64300fb78bc28064bed9c653931`.

Independent review: https://github.com/amdpilot-org/sglang/pull/2621.

The candidate's useful benchmark note and distinction between client and engine
windows are preserved. Its statement that TPOT and ITL cover the whole request
lifetime is corrected:

- TPOT excludes TTFT and measures the post-first-token decode phase.
- ITL samples are intervals between nonempty stream events. Event batching means
  they need not be one sample per token, and the terminal response tail can be
  absent from their sum.
- `Output token throughput` covers the whole benchmark run, but it remains a
  different estimator from the engine's recent reporting-window throughput and
  is not expected to agree exactly.

## Evidence

`evidence/candidate/adversarial-metrics-confirmed.txt` records a deterministic
candidate run with 100 ms TPOT versus 490 ms whole-request time per token, eight
ITL samples for nine post-first tokens, a 0.8 s ITL sum versus a 0.9 s decode
phase, and 2.04 tok/s whole-run output throughput versus a valid 20 tok/s recent
window.

`evidence/candidate/corrected-regression-before.txt` is the failing-before test
against the exact candidate (two failures). `evidence/after/corrected-regression-after.txt`
is the passing-after result. `evidence/tests/source-formulas.txt` records the
actual source locations for the client and engine calculations.

## Limitations

No GPU execution was required for this correction because the counterexamples
exercise deterministic metric arithmetic and output wording. The original
Qwen2.5-0.5B weights and NVIDIA H800/CUDA environment were unavailable, so this
work does not claim reproduction of that model/hardware workload or compare its
absolute performance. No native source changed and no native rebuild applies.
