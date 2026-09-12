# Independent review of PR 2738

Candidate: `1a422dd3f344dd098af11e25a4bad951bd9daf62`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **request changes**. The candidate is a useful partial implementation, but it does not fully resolve the original feature request.

## Findings

1. The startup resolver trusts the `feasible` bit in a report and never validates the selected plan against current capabilities. It also does not pass a current code revision to `resolve_calibrated_plan`, and the report's environment is stored but never checked. The independent adversarial case supplied an exact-signature report marked feasible but with stale code/environment and `num_gpus=1` alongside CFG/TP/SP/DP degrees of 2 and FSDP enabled. `apply_advisor_to_server_args` selected and applied that impossible plan. This violates the central requirement to select a legal topology and the requested fail-closed handling of stale code/device records. See `adversarial.log`.

2. The candidate adds schema types, enumeration helpers, exact-signature matching, quality filtering, p95 ranking, explicit-flag precedence, startup arguments, and focused unit tests. Its submitted regression command passes (210 tests and 67 subtests). These are meaningful partial feature components, not merely test-only hardening.

3. Major original-contract components remain absent: an isolated calibration command/runner; complete warmup and repeated end-to-end measurement; cold-start separation; peak-memory, stage-time, and collective count/byte collection; model accuracy/trajectory gate execution; pipeline capability methods; startup metrics; and routing between separately launched pools. The world-size-one candidate test verifies only argument assignment. The no-op collective/device-communicator behavior it cites already exists on the recorded base; the candidate adds no execution-path assertion that a serving model avoids sharding wrappers and redundant broadcasts.

4. The one available AMD Instinct MI355X was detected with Torch 2.11.0+rocm7.2 and HIP 7.2. No diffusion model weights, model-specific quality reference, or multi-GPU allocation were available. Consequently no end-to-end diffusion calibration matrix, multi-rank topology initialization, or independent semantic/numerical quality gate could be qualified. No GPU smoke is counted as proof of the feature.

## Source and native paths

At the exact candidate checkout, Python imported the changed modules from `/job/repo/python/sglang/multimodal_gen/runtime/server_args/parallel_advisor.py` and `/job/repo/python/sglang/multimodal_gen/runtime/server_args/server_args.py`. Torch imported from `/opt/venv/lib/python3.12/site-packages/torch/__init__.py`. The candidate changes only Python, documentation, tests, and report artifacts; it changes no native source, so no native rebuild was applicable. An existing AITER cache library was imported incidentally by the test environment and is not evidence for this feature.

## Classification

This is a **partial fix**. It is not a full original-issue fix, not test-only hardening, and not an entirely unverified claim. The submitted unit behavior is verified, while the core legality/staleness counterexample fails and the end-to-end calibration and distributed/model-specific portions remain unverified or unimplemented.
