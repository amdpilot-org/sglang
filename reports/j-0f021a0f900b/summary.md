# Independent review of amdpilot-org/sglang PR 1797

Reviewed exact commit `fd0ffce08d62f7556170a4603a23548c10e8b726` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

The earlier candidate `aae252eb457f1f0358ae4b98897f3a8fa6eda50d` independently reproduced the reported divergent-duplicate defect: after sequence 0 admitted a RID, a second sequence 0 rejecting the same RID was logged and silently ignored. The exact reviewed candidate raises `RuntimeError` before a second queue application. It also preserves immutable forwarding, detects sequence gaps/old replays, orders aborts after frozen decisions, routes local KV failures through later consensus, reserves metadata before reporting good, and includes HiCache readiness in consensus.

The candidate contains a substantive PD-router change, not test-only hardening: retryable responses cause another attempt only when tagged `RetryableBeforePDDispatch`; failures returned after `execute_dual_dispatch_internal` are not tagged and are therefore not replayed. Its regression asserts a failing decode worker sees exactly one generate request despite retries being configured.

Recommendation: **unverified**. The Python protocol correction is verified by the candidate suite and independent adversarial tests, and no remaining deterministic Python counterexample was found. However, this image has no `cargo`/`rustc`, so the Rust router change could not be compiled or exercised, and the single gfx950 cannot reproduce the original PP=8 Mooncake cancellation-storm topology. Accordingly, this review does not claim the complete production issue is fully resolved.

Raw commands, outputs, source import paths, issue/PR snapshots, and the independent adversarial script are retained in `evidence/`.
