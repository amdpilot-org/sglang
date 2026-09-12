# Independent review of PR 2548 at ed2ca323422f819e95fc8be3f3494df036d12b09

Upstream issue: https://github.com/sgl-project/sglang/issues/31861

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2525

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2553

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2548

## Finding

Request changes. The candidate is a partial correctness fix, not complete FP32
MoE-router coverage. It fixes the reported DeepSeek AMX-selected BF16-output
path and the previously identified Ernie4 counterexample, and it enforces an
FP32 input boundary at `biased_grouped_topk_cpu`. Its seven focused tests pass.

An independent adversarial check found a remaining caller on the prepared
source: `Dots3MoEGate` uses BF16 `F.linear` for its non-tiny path. At the exact
candidate commit it emits two BF16 logits `[11.75, 11.75]`, while an independent
FP32-operand reference gives `[11.780056953430176, 11.775662422180176]`. The
candidate's new TopK guard then raises `ValueError` before native dispatch.
Therefore the candidate does not fulfill the original issue's general router
contract that CPU/AMX gating logits reach TopK as FP32.

The optional specialized BF16-activation x FP32-weight AMX kernel remains
unimplemented. That is a performance gap rather than a correctness failure
where the candidate uses FP32 `F.linear` fallback.

## Evidence

- `evidence/failing-before-base.txt`: recorded base
  `358c163250ad3b1f62939b01ce1314a0a31a0365` emits BF16 for both the DeepSeek
  AMX-selected fixture and Ernie4, and the base CPU TopK accepts Ernie's BF16
  logits.
- `evidence/candidate-regression.txt`: candidate regression module, 7 passed.
- `evidence/candidate-focused-probe.txt`: exact candidate makes DeepSeek and
  Ernie4 emit FP32; packed linear is not called for the DeepSeek CPU fixture.
- `evidence/candidate-adversarial-dots3.txt`: remaining BF16 Dots3 logits and
  rejection by the candidate's FP32-only TopK guard.
- `evidence/gpu-linear-reference.txt`: on the assigned AMD Instinct MI350X,
  the existing GPU BF16 x FP32 router linear returned FP32 and agreed with an
  independent CPU FP32 reference (maximum absolute error
  `2.5033950805664062e-06`). This is a reference-path check, not AMX evidence.
- `evidence/import-paths.txt`: Python model and TopK imports resolve to the
  candidate checkout. `sgl_kernel` resolves to the prepared installed egg.
- `evidence/native-changes.txt`: empty; the candidate changes no native source,
  so a native rebuild is not applicable.

## Environment limitations

The host is an AMD EPYC 9965 system and has no Intel AMX CPU. Actual AMX native
execution, ISA use, and performance could not be tested. No affected full-model
weights were available, so this review does not claim full-model, serving-path,
or distributed validation. The available GPU is AMD Instinct MI350X under
PyTorch 2.11.0+rocm7.2; it can validate the GPU numerical reference but cannot
validate Intel AMX behavior.
