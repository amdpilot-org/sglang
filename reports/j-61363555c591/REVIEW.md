# Independent review of PR 1749

Reviewed candidate: `bed37704472eae98455502c1b03037640481407c`

Upstream issue: https://github.com/sgl-project/sglang/issues/33740

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1784

Recommendation: **accept**. The candidate fully resolves the benchmark-side contract in the original issue.

## Evidence

The recorded base selected `/v1/models` as the readiness URL for SGLang. Against a controlled router double where `/v1/models` returned 200 while `/health_generate` returned 503, the base declared the server ready immediately and advanced into model/tokenizer setup. Its observed paths were `/v1/models`, `/v1/models`, and `/model_info`.

At the exact candidate commit, the identical command made only `/health_generate` requests and exited after the configured readiness timeout. This is the intended behavior while a prefill worker is still unavailable.

The source contract was also checked rather than inferred from the candidate prose. `sgl-model-gateway/src/routers/http/pd_router.rs` implements PD `/health_generate` by selecting a prefill/decode pair, probing both workers' `/health_generate` endpoints concurrently, and returning success only when both probes succeed. Therefore the candidate's benchmark endpoint change is tied to generation readiness of both PD sides.

The prepared interpreter imported `sglang` and `sglang.benchmark.serving` from `/job/repo/python`, confirming that the tested source was the checkout rather than an installed copy. The focused candidate tests passed with 2 tests and 10 subtests; the containing module passed with 47 tests and 16 subtests.

## Classification and limitations

This is a source fix plus regression coverage, not test-only hardening or an unverified claim. No C++/native files changed, so no native rebuild was applicable.

The original GLM-5.2-FP8 weights and two H200x8 nodes were unavailable. The prepared environment has one gfx950 GPU with ROCm 7.2, so this review does not claim a full CUDA/H200 TP8 multi-node Mooncake reproduction, model accuracy validation, or distributed transport qualification. Those architecture limitations do not leave a known counterexample to the narrow HTTP readiness contract.
