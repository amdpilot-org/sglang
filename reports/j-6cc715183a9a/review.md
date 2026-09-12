# Independent review of PR 2524

Upstream issue: https://github.com/sgl-project/sglang/issues/28312

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2470

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2544

Candidate: https://github.com/amdpilot-org/sglang/pull/2524 at
`12e81109f0ecb3fa6bf480acbcde8b15c6589d1b`

Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
(there was no difference between the recorded base and the initial checkout).

## Recommendation

Accept as test-only hardening. The candidate does not change the runtime fix:
the recorded base already contains upstream PR 27998's `None`-to-slot-0 guard.
The added tests accurately cover the reported exception boundary and useful
mixed/override cases. Removing only that existing guard reproduced the exact
`TypeError`, while the candidate's three tests and independent GPU cases passed.

This is not an independently verified full resolution of the original serving
incident. The candidate and this review exercise the precise helper contract,
not Qwen3.5-397B-A17B-FP8 serving on two NVIDIA B300 GPUs under production
traffic. The original issue also remains open. Accordingly,
`fully_resolves_original` is false even though no defect was found in the
candidate's test coverage or in the current helper behavior.

## Evidence

- `raw/failing_before_independent.log`: on the prepared base, temporarily
  removing only the existing normalization guard makes a mixed `[None, 1]`
  request fail in `torch.tensor` with the reported `TypeError`. The tracked
  source was restored byte-for-byte before switching revisions.
- `raw/candidate_pytest.log`: the exact candidate's regression suite passed,
  3 tests in 8.55 seconds.
- `raw/candidate_adversarial_gpu.log`: the exact candidate source was imported
  from `/job/repo/python/sglang/srt/managers/schedule_batch.py` and executed on
  one AMD Instinct MI355X (`gfx950`). Non-identity request-pool ordering with
  `[None, 1, None]` selected `[400, 101, 300]`; explicit `[1, 0, 1]` selected
  `[401, 100, 301]`, both exactly matching independent references.
- `raw/base_imports_and_gpu.log`: records the prepared interpreter, source
  import path, Torch/ROCm versions, device, ISA, and inspected helper source.

No C++/CUDA/HIP/native source differs between the base and candidate. Therefore
no native rebuild was required or performed; the tested implementation was the
repository Python source, not an unrelated installed copy.

## Remaining limitations and counterexamples

- Qwen/Qwen3.5-397B-A17B-FP8 weights were unavailable.
- The assigned system has one AMD gfx950 GPU with ROCm 7.2, not two NVIDIA B300
  GPUs with CUDA 13 and TensorRT-LLM attention.
- TP=2, the reported HTTP serving configuration, long-running production
  traffic, prefix-cache history, and semantic model output were not exercised.
- The synthetic helper fixture establishes normalization and gather behavior;
  it does not prove that every real scheduler state reaching Spec-v2 verify has
  a semantically usable slot-0 mapping.

