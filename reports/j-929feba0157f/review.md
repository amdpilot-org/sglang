# Independent review of PR 1335

Reviewed candidate commit `94025bba2e2c5da8b438a0e0788c4a196798ee72` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the open issue contract.

Recommendation: **accept**.

The recorded base reproduced the policy defect with the candidate regression: a real compressed-tensors-style object whose `ignore` contains individually named `mtp.*` weights was returned unchanged by `_mtp_quant_config`. The test failed at the expected assertion. At the exact candidate commit, the same regression passed.

Independent checks used the real `CompressedTensorsConfig`, not only the candidate's `SimpleNamespace`. The complete individually listed MTP ignore set from the report and a single real `mtp.fc` entry both disabled quantization. Empty ignores and lookalikes (`mtpfoo.fc`, `model.mtp.fc`) retained quantization. This confirms the changed branch is reached through the actual config type and is bounded to literal `mtp.` prefixes. The candidate passes `None` into the existing MTP model constructor, so fused `qkv_proj` and `gate_up_proj` are constructed unquantized, matching the plain BF16 checkpoint tensors described by the issue.

No native source changed, so a native rebuild was not applicable. Imports resolved to `/job/repo/python/sglang` and Torch resolved to `/opt/venv/lib/python3.12/site-packages/torch` through the prescribed interpreter.

## Limitations

The reported Qwen3.8 27B compressed-tensors weights were unavailable. The prepared host has one AMD Instinct MI355X (`gfx950`, ROCm 7.2), rather than the report's two RTX 3090 CUDA devices. Therefore this review did not independently reproduce TP=2 serving, skipped-weight log counts, acceptance length, throughput, or NVIDIA behavior. No GPU kernel execution was needed for this Python construction-policy change, and GPU availability alone is not claimed as GPU validation.

Compressed-tensors regex ignores such as `re:^mtp\\..*` do not trigger the new literal-prefix branch. That is outside the original issue's stated contract, which specifically uses individually listed `mtp.*` layer names, but is retained as a documented boundary rather than presented as tested support.

Raw JUnit evidence is included beside this report.
