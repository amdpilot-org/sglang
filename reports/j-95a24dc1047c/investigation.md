# Independent review of amdpilot-org/sglang#1146

Candidate reviewed: `73e1766e4e1351a402fbd74bb9acccda53e6d9f1`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/37912

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1179

Parent correction PRs: https://github.com/amdpilot-org/sglang/pull/923 and https://github.com/amdpilot-org/sglang/pull/1041

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue's three concrete gateway gaps in the tested source paths:

1. The actual `openai_protocol::responses::ResponsesRequest` accepts all seven values (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
2. Responses-to-chat conversion forwards every accepted effort string.
3. Harmony maps `none`/`minimal`/`low` to Low, `medium` to Medium, and `high`/`xhigh`/`max` to High in both typed Responses and string Chat inputs.

This is a source fix plus regression hardening, not a test-only change. Cargo resolved `openai-protocol v1.0.0` from `sgl-model-gateway/third_party/openai-protocol` at the candidate, proving the tested build did not silently use the original four-variant registry package.

## Independent evidence

On the recorded base, a temporary revision-neutral Rust example deserialized the actual gateway request type from the seven request bodies. It accepted only `minimal`, `low`, `medium`, and `high`; `none`, `xhigh`, and `max` failed with Serde `unknown variant` errors before handler execution.

The identical probe at the candidate accepted all seven. Independent boundaries also behaved consistently: unknown `turbo` and uppercase `HIGH` were rejected, while a null effort, omitted effort, and omitted reasoning object were accepted.

Candidate-focused tests passed:

- `cargo test harmony_effort -- --nocapture`: 2 passed.
- `cargo test reasoning_effort -- --nocapture`: 2 passed.
- `cargo test --lib`: 396 passed.
- `cargo fmt --all -- --check`: passed (stable rustfmt emitted warnings for nightly-only formatting options).
- `git diff --check 358c163...73e1766...`: passed.

The temporary probe was removed before returning to the prepared branch. Raw logs were preserved outside the revision-switched checkout under `/job/review-evidence-j-95a24dc1047c/`.

## Environment and limitations

The review ran on Linux x86_64 with Rust 1.90.0, using a private Cargo target directory under `/tmp/amdpilot-repo-j-95a24dc1047c`. The prepared Python environment reports Torch 2.11.0+rocm7.2 and HIP 7.2, but GPU execution was not used: request deserialization and Rust gateway conversion/clamping are CPU-only code paths, and no model architecture or weights are required to verify this contract.

No C++/FlyDSL/native source changed, so no native rebuild was applicable. A live model-backed HTTP server was not started; therefore this review does not claim model semantic accuracy, architecture coverage, or distributed serving validation. Those are unrelated to the original pre-handler deserialization and gateway mapping defect.

Remaining counterexamples: none found within the original issue contract.
