# Independent review of PR 739

Reviewed exact candidate commit `a2979819b4ac3df4fea4e93f41523a4efdc7a841` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully resolves the original issue for the covered dense FP8 linear path.

## Evidence

- The recorded base reproduced the incorrect capability result for SM75/SM80 and propagated a forced unsupported `torch._scaled_mm` error from scalar per-tensor dispatch and the nominal fallback.
- The exact candidate returned the expected capability matrix: SM75 false, SM80 false, SM89 true, SM90 true.
- Independent adversarial tests forbidding `torch._scaled_mm` passed through `apply_fp8_linear` with scalar scales and through the direct fallback with scalar and channelwise scales, including the single-token case. Results matched explicit FP32 dequantization plus `torch.mm`.
- The candidate regressions passed: two focused fallback tests and the platform capability test with four subtests.
- The quantization documentation now states the SM89+ native FP8 GEMM requirement and distinguishes it from FP8 storage/conversion.
- The changed software fallback executed successfully on the assigned AMD Instinct MI350X and matched the independent reference.

## Scope and limitations

There was no physical NVIDIA GPU, so CUDA architecture qualification uses controlled capability probes and forced unsupported-kernel dispatch rather than real SM75/SM80 hardware. `supports_fp8()` remains unused by production code, but the original issue permits either early refusal or a numerical software fallback; the candidate fixes the actual execution path with the latter. The fallback is correctness-oriented and converts operands to FP32, so it has performance and memory costs.

The candidate changes no native source, and the prepared environment records no native build target, so no native rebuild was applicable. An independent attempt to exercise the unchanged ROCm native `_scaled_mm` path encountered a hipBLASLt solution limitation for the selected probe shape; the changed software fallback itself passed on MI350X.

Raw commands and outputs are retained under `reports/j-e89d63e91ed9/raw/`.
