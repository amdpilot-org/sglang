# Independent review of PR 3027

Reviewed exact commit `3559565b2ea09da7c55cad83ee909bf763d8f0d9` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **unverified**. The candidate directly addresses the source-level draft exclusions and its state-isolation and alignment tests pass. It should not yet be described as a full original-issue fix because the motivating DeepSeek MTP configuration at EP 72/144 was not executable in this environment.

The base reproduces the issue at the implementation level: `ModelRunner.maybe_init_expert_location_metadata()` returns for every draft worker, and `maybe_init_eplb_manager()` excludes drafts. The candidate removes those exclusions, keeps target and draft EPLB resources separate through `speculative_moe_a2a_backend_context()`, and pads the draft's physical expert count independently.

The candidate regression passed, as did all 35 existing EPLB unit tests. Its one-GPU script produced an exact NumPy match for a synthetic 256-to-288 expert mapping. That script is useful arithmetic/mapping evidence, but it does not exercise SGLang's EPLB manager, communications, model loading, rebalancing, or speculative serving execution.

The registered EPLB serving test was attempted and failed before engine execution because `lmsys/sglang-ci-dsv3-test` returned HTTP 401. Only one AMD Instinct MI355X is visible. DeepSeek/MTP weights are unavailable. Consequently, EP 72/144 startup, live draft rebalancing, cross-rank migration, and semantic correctness remain unverified.

No native source changed. A native rebuild was therefore not applicable. Tests imported SGLang from the exact checkout and loaded the prepared pinned AITER artifact.

Raw command output and fetched issue/PR metadata were preserved outside the checkout at `/job/review-evidence-j-bf100ac0661b` while revisions were switched.
