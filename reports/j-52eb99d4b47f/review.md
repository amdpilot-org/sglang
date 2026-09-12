# Independent review of PR 3374

Upstream issue: https://github.com/sgl-project/sglang/issues/34023

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3339

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3376

Candidate: https://github.com/amdpilot-org/sglang/pull/3374 at exact commit
`41a98abfdc81dee12010be10ba4805638645f7ff`.

## Verdict

Recommendation: **accept**. The candidate fully resolves the narrow original
contract at the unit/numerical level: compact confidence computation now derives
the active draft window from `x_post_hc`, and its Markov context uses that same
runtime window instead of checkpoint-native `self.gamma`.

The recorded base and image-prepared checkout were identical at
`358c163250ad3b1f62939b01ce1314a0a31a0365`. On that base, checkpoint gamma 5
accepted runtime gamma 5 but raised the reported invalid reshape for runtime
gamma 1, 3, and 7. At the exact candidate commit, its regression passed, and an
independent GPU matrix covering batch sizes 1/2/4, runtime gamma 1/2/3/6/7/9,
and both Markov and no-Markov branches matched a separately computed tensor
reference exactly (`max_abs_error=0.0`). A malformed tensor whose token count
was not divisible by the batch size remained rejected.

The changed implementation and tests are Python-only. No C++, CUDA/HIP,
FlyDSL, or other native source changed, so no native rebuild was applicable.
Imports resolved to the checked-out source at
`/job/repo/python/sglang/srt/models/deepseek_v4_dspark.py`, not to a separately
installed copy.

## Scope and limitations

The assigned accelerator was one AMD Instinct MI350X (`gfx950`), not the
reporter's four RTX PRO 6000 (`SM120`) GPUs. DeepSeek-V4-Flash-0731 weights were
not available. Therefore this review does not independently claim full model
loading, CUDA decode-graph capture, HTTP serving, semantic generation quality,
multi-GPU behavior, or the reported SM120 corruption screen. Those are
architecture/model validation limitations, not counterexamples to the focused
reshape and Markov-window fix.

No remaining counterexample was found within the executable original contract.
The candidate's regression itself covers runtime gamma 3, 5, and 7; the
independent matrix adds gamma 1, 2, 6, and 9, multiple batch sizes, and the
no-Markov branch.

## Raw evidence

- `base-reproduction.log`: failing-before behavior at the exact recorded base.
- `candidate-regression.log`: candidate-authored regression, 3 tests passed.
- `candidate-adversarial-gpu.log`: source/import paths, accelerator identity,
  independent numerical matrix, and malformed-shape rejection.

