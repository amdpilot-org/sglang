# Independent candidate review evidence

Candidate reviewed: https://github.com/amdpilot-org/sglang/pull/2301 at exact commit `b4d9051eff6bffcbf5d0c00500cebdd09e924267`.

Upstream issue: https://github.com/sgl-project/sglang/issues/31459

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2237

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2336

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue's deterministic routing contract on the recorded base. It is a source fix with regression hardening, not a test-only change or an unverified claim.

The source change preserves an existing initial `tool_choice` (including structured values), defaults an omitted choice to `"auto"`, and explicitly sets the post-MCP resume payload to `"auto"`. Both streaming and non-streaming loops use the corrected helpers.

## Revisions and source paths

- Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- Prepared checkout initially matched that base exactly on `amdpilot/j-92454e6359ea`.
- Exact candidate: `b4d9051eff6bffcbf5d0c00500cebdd09e924267`.
- Review returned to `amdpilot/j-92454e6359ea` before this report was committed.
- Source under test: `/job/repo/sgl-model-gateway`.
- Job-private Rust/Cargo homes: `/tmp/amdpilot-repo-j-92454e6359ea/{rustup,cargo}`.
- Rebuilt Rust test artifacts: `/tmp/amdpilot-repo-j-92454e6359ea/cargo-target`.
- `origin/main` had advanced to `bd45cd50ca900dd821f829ca9adfbf9aa3336bda` during review; the mandated recorded base remains its ancestor. This does not alter the failing-before comparison.

The candidate changes Rust source/tests only. It changes no C++, FlyDSL, Python extension, or other native-library source, so no separate native rebuild/import path exists. The gateway and tests were compiled directly from the checked-out candidate source rather than using a wheel.

## Failing-before reproduction

Candidate test instrumentation was temporarily applied without the candidate source fix to the exact recorded base. Both real gateway integration paths failed on the initial forwarded JSON value:

```text
cargo test --test api_tests api::responses_api_test::test_non_streaming_mcp_minimal_e2e_with_persistence -- --exact --nocapture

assertion `left == right` failed: initial request should preserve the client's tool_choice
  left: Some("auto")
 right: Some("required")
test result: FAILED. 0 passed; 1 failed; 86 filtered out
```

```text
cargo test --test api_tests api::responses_api_test::test_streaming_with_mcp_tool_calls -- --exact --nocapture

assertion `left == right` failed: initial request should preserve the client's tool_choice
  left: Some("auto")
 right: Some("required")
test result: FAILED. 0 passed; 1 failed; 86 filtered out
```

The streaming base run still completed the MCP call and produced the final assistant message, isolating the failure to the original `tool_choice` forwarding contract.

## Candidate validation

At the exact candidate commit:

- Non-streaming integration regression passed and captured exactly two mock-upstream requests: initial `required`, then resume `auto`, with `function_call_output` in the resume and a completed MCP call/final assistant response.
- Streaming integration regression passed with the same two-request sequence and final assistant text `Tool result consumed; here is the final answer.`
- Candidate unit boundaries passed for explicit `required`, `auto`, and `none`, plus omitted choice defaulting to `auto`.
- `cargo fmt -- --check` and `git diff --check BASE CANDIDATE` passed.

Independent review-only tests, temporarily added and removed at the candidate revision, also passed:

- A structured `allowed_tools`/`required` choice survived the initial transformation byte-for-value while an unrelated `temperature` field remained unchanged.
- `build_resume_payload` changed that structured required choice to `"auto"`, retained unrelated fields, appended tool history, and set the requested streaming/store flags.

Raw command logs and revision snapshots were preserved outside the checkout under `/tmp/amdpilot-repo-j-92454e6359ea/` while revisions were switched.

## Related changes and limitations

The original issue remained open during review. The related upstream PR https://github.com/sgl-project/sglang/pull/31469 was also open at commit `1314fa761f6872e4f1c63912adaca57c91a78b09`; its source direction is consistent with this candidate.

Host architecture was x86_64 Linux. One AMD Instinct MI355X (`gfx950`, ISA `amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-`) was visible but unused because the defect and tested contract are entirely in Rust JSON routing and mock HTTP/MCP execution. No model weights, real external MCP service, semantic model behavior, multi-node workload, or GPU kernel was exercised or claimed. Those are not required to establish this payload-rewrite bug, but remain environment-scope limitations.
