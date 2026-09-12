# Independent review of amdpilot-org/sglang PR 825

Candidate reviewed exactly at `e2d7a142c48233c30f194b4f16ca5d460bb805c5` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The candidate is a partial fix. Its focused regression suite passes and it closes the previously demonstrated URL image-load and direct `process_mm_data` override gaps, but it does not fully enforce `sglang:mm_processor_seconds` around HF processor work performed by specialized async implementations.

## Blocking counterexample

`TransformersAutoMultimodalProcessor.process_mm_data_async` invokes `_apply_hf_processor` directly on its non-audio path. The candidate's `__init_subclass__` wrapper only decorates `process_mm_data`, so this real implementation bypasses the processor observation entirely.

The independent test in `raw/test_adversarial_async_bypass.py` runs the actual async override with a bounded 20 ms stub for `_apply_hf_processor`. The method returns the expected output, but the recording collector receives zero processor observations (`assert 0 == 1`). See `raw/candidate-adversarial.txt`.

## Evidence

- Recorded base: the candidate's unchanged regression file failed 7/7, including the original URL observation and async override failures. See `raw/base-regression.txt`.
- Exact candidate: its regression suite passed 7/7. See `raw/candidate-regression.txt`.
- Exact candidate, independent adversarial case: failed because the TransformersAuto direct HF processor path emitted no processor timing. See `raw/candidate-adversarial.txt`.
- Imports resolved to `/job/repo/python/sglang/...`, confirming the checkout source was exercised.
- The diff contains Python and report/test changes only. No C/C++/CUDA/HIP/native source changed, so a native rebuild was not applicable.

## Scope and limitations

The deterministic tests cover CPU-side media loading and processor timing without model weights. GPU execution was not needed for the failing contract and was not performed. Therefore this review does not qualify model-specific semantic accuracy, GPU kernels, distributed serving, or unavailable model architectures. Those limitations do not weaken the reproduced counterexample, which occurs before model execution in the tokenizer-manager multimodal preprocessing path.

Upstream issue: https://github.com/sgl-project/sglang/issues/38676

Mirror issue: https://github.com/amdpilot-org/sglang/issues/864

Candidate PR: https://github.com/amdpilot-org/sglang/pull/825
