# Independent review of PR 920

Reviewed https://github.com/amdpilot-org/sglang/pull/920 at exact commit
`597898a956a8a2fba24d32ecf0ef6c32b9666a6b` against the original issue.

Recommendation: **request changes**. The candidate is a useful partial fix: it
makes the construction failure actionable, corrects the ignore guidance, and
does not silently allocate dense parameters for a possibly packed MTP layer.
It does not resolve the original no-ignore startup failure.

The recorded base and the exact candidate were both exercised with a real
`CompressedTensorsConfig` and `ReplicatedLinear`. On the base, the no-ignore
case raised the generic error. On the candidate, the same case still raises,
but now includes the MTP prefix, matched `Linear` target, pack-quantized format,
W4/A8 arguments, and safe ignore guidance. The candidate's own regression
explicitly expects that failure.

Valid `re:` ignores select `UnquantizedLinearMethod` for both `mtp...` and
`model.mtp...` prefixes. Plain `mtp.*` and `model.mtp.*` entries do not match
children. One MI355X GPU run validated the explicit-ignore BF16 path against a
CPU FP32 reference. Actual Qwen3-Next dense and packed checkpoint loading and
the original 2x H100 TP2/EP2 serving configuration remain unverified.

Upstream issue: https://github.com/sgl-project/sglang/issues/38574

Mirror issue: https://github.com/amdpilot-org/sglang/issues/952
