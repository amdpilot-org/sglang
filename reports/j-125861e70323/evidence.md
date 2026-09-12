# Independent review evidence: PR 3019

- Candidate: `5eec3ac5f2f27be36c9c785b36c6071cb7297155`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Review checkout restored to: `amdpilot/j-125861e70323`
- Upstream issue: https://github.com/sgl-project/sglang/issues/6622
- Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2967
- Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3056

## Finding

Recommendation: **accept**. The exact candidate completes the original configuration contract on top of related work already present at the recorded base. The base already had a 512 MiB default, `--max-payload-size`, and both Axum/tower enforcement layers, so a 10 MiB request was no longer a failing case there. The reproducible base defect was the missing environment path: with `SGLANG_MAX_PAYLOAD_SIZE=12582912`, the Python parser still returned `536870912`. The candidate returned `12582912`, while an explicit CLI value returned `16777216`.

This is a full fix for the open issue as it exists at the recorded base, but only part of the implementation is introduced by this candidate. It is not a claim that this single commit created the pre-existing CLI or enforcement layers.

## Source and native paths

- Native CLI environment parsing: `sgl-model-gateway/src/main.rs`
- Python launcher parsing: `sgl-model-gateway/bindings/python/src/sglang_router/router_args.py`
- HTTP enforcement: `sgl-model-gateway/src/server.rs`
- Native regression: `sgl-model-gateway/tests/routing/payload_size_test.rs`
- Test upstream: `sgl-model-gateway/tests/common/mock_worker.rs`
- Rebuilt executable: `/tmp/amdpilot-repo-j-125861e70323/gateway-target/debug/sgl-model-gateway`
- Full raw logs and checksums: `/tmp/amdpilot-repo-j-125861e70323/review-evidence/`

The executable was independently rebuilt with Rust 1.90.0. `file` identified it as an x86-64 ELF PIE; the candidate changes no GPU or FlyDSL native code.

## Results

- Native environment/precedence regression: 3 passed.
- Gateway payload regression: 5 passed, including exact configured limit, above-limit rejection, and a 3 MiB body accepted with a 4 MiB gateway limit.
- Prepared Python source import resolved to `/job/repo/sgl-model-gateway/bindings/python/src/sglang_router/router_args.py` at the candidate revision.
- The full Python test module could not collect because the prepared environment has no `sglang_router.sglang_router_rs` extension. Direct parser checks passed for a 10 MiB environment value, whitespace, explicit CLI precedence, and the prefixed launcher form.
- The native executable rejected a non-integer environment value with exit code 2.
- Python argparse accepts zero, negative, and arbitrarily large integers, as its pre-existing CLI path already did. Conversion/config validation occurs later. This is a validation-consistency limitation, not a counterexample to increasing the limit with a valid positive byte value.

## Architecture and environment limits

No GPU was used. Request admission and forwarding are CPU HTTP behavior before model execution, and the patch contains no numerical or architecture-specific GPU change. No multimodal model weights were available or required to establish the gateway byte boundary; therefore this review makes no image-model semantic-accuracy or distributed-workload claim. The missing Python native extension prevented an end-to-end Python-binding launch, while the independently rebuilt standalone native gateway covered the changed Rust path and transport enforcement.
