# Independent review of PR 3515

Reviewed exact candidate `b1bb73d26a50ba4a052cad080e9aa1aad0f167ae` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept**. The candidate fully resolves the original single-tokenizer in-process Engine-bound ASGI factory request.

## Findings

- The prepared checkout exactly matched the recorded base.
- The base reproduced the original failure: `build_app` and `init_app_state` could not be imported.
- The candidate adds fresh application construction, binds tokenizer/template/scheduler state on each app without publishing it through the compatibility global, and routes global-style endpoint lookups through request-local context.
- The single-tokenizer launch path serves an app created by `build_app`; uvicorn, SSL-refresh, and embedded Granian branches all receive that instance.
- The legacy module app and multi-tokenizer bootstrap remain available.
- The candidate's 31 focused tests plus 12 subtests passed.
- Independent concurrent mounted-app tests passed for `/model_info`, `/generate`, and `/v1/models`; each response used the correct manager while a conflicting legacy global remained unchanged.
- A real Engine using the qualified deterministic tiny Llama fixture served `/model_info` and executed `/generate` through the factory app on GPU, producing eight tokens. Engine shutdown reaped both owned worker descendants.

## Classification

This is a full original-issue fix, not a partial fix, test-only hardening, or an unverified claim. No remaining counterexample was found within the approved single-tokenizer in-process scope.

## Environment and limits

The GPU was an AMD Instinct MI355X (`gfx950`) with Torch `2.11.0+rocm7.2` and HIP `7.2.26015`. Qualification covered one synthetic Llama on one GPU. It does not establish semantic model quality, other architectures, distributed execution, multi-tokenizer hosting, HTTP/2, TLS refresh, or native gRPC behavior. Process-wide SGLang configuration still means multiple in-process apps must use compatible configuration. No native source changed, so native rebuild was not applicable.

Raw logs and scripts were preserved outside the checkout at `/tmp/amdpilot-repo-j-3b11e8e0bec6/review-evidence` while revisions were switched. Structured commands and measured results are in `result.json`.
