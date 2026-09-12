# Independent review of PR 1165

Candidate: `c8db9fb4e34369abffa08047daaa9418e610761a`

Upstream issue: https://github.com/sgl-project/sglang/issues/36764

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1199

## Verdict

Accept. The candidate is a source-level fix for all three transient paths measured in the original issue's primary `docker/Dockerfile`. The recorded base is the candidate's direct parent. On that base, the candidate regression fails because all three destinations are committed by `COPY` and removed in later `RUN` layers. At the exact candidate commit, the focused suite passes, and an independent logical-instruction check confirms each consumer is coupled to the correct BuildKit bind mount with no `COPY` of the transient destination.

This is not merely test-only hardening: the production Dockerfile changes remove the three layer-producing `COPY` instructions. Persistent builder outputs remain copied normally.

## Evidence

- Base regression: 1 failed, 2 passed; the failure reports `/tmp/wheels/hpc-ops`, `/tmp/local_src`, and `/tmp/gateway_wheels`.
- Candidate focused tests: 10 passed, including the new regression and existing Docker build metadata tests.
- Independent contract check: all three transient destinations have no `COPY`; each appears in one logical `RUN` containing both its exact builder bind mount and its consumer command. Retained flashinfer and gateway binary outputs still use `COPY`.
- `git diff --check` passes.

## Limitations and remaining related patterns

Docker, Podman, and `buildctl` are absent from the prepared environment. Therefore no complete image build, registry pull, compressed-layer remeasurement, or runtime inspection was possible. The conclusion is based on the checked-out Dockerfile and Docker/BuildKit layer semantics, not newly measured image bytes.

GPU execution is irrelevant to this packaging-only defect. The candidate changes no Python import path or native source, so no source/native import-path override or native rebuild applies. The prepared environment is x86_64 with ROCm Torch available, but neither GPU architecture nor model weights affect this review.

The unmeasured analogous patterns explicitly listed by the reporter remain in `docker/Dockerfile.cu134` and `docker/rocm.Dockerfile`; `docker/gateway.Dockerfile` also remains outside this candidate. They are related counterexamples for other image definitions, but not counterexamples to the three quantified `lmsysorg/sglang:latest` paths in the primary Dockerfile.
