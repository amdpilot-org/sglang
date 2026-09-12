# Independent review of PR 2941

Candidate: https://github.com/amdpilot-org/sglang/pull/2941

Exact commit: `958eda57e655d02d2ac39165dc77610ab240b31a`

Upstream issue: https://github.com/sgl-project/sglang/issues/18427

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2892

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2973

## Recommendation

Accept.

The candidate's two-line redirect correction is semantically justified by an independent source: immediately before the original migration, `docs/basic_usage/qwen3.mdx` was titled “Qwen3-Next Usage” and exclusively described Qwen3-Next models. The recorded base incorrectly routes that legacy URL to the generic Qwen3 cookbook page. The exact candidate routes it to the existing Qwen3-Next cookbook page in both the checked-in configuration and generator.

No remaining counterexample was found for the static redirect contract.

## Scope classification

This candidate is a **partial follow-up fix plus regression hardening**, not a standalone implementation of the full original issue. The full removal/migration work from upstream PR 25813 was already present at recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`:

- the legacy popular-model index and model pages were absent;
- `/basic_usage/popular_model_usage.html` already redirected to `/cookbook/autoregressive/intro`;
- all 13 reviewed legacy routes had existing cookbook destinations.

Consequently, the original pre-migration failure cannot be reproduced on the required recorded base. The narrower Qwen3 misroute does fail on that base and passes on the exact candidate. The resulting candidate tree fully satisfies the original repository-level removal and redirect contract, but most of that result comes from pre-existing base work.

## Independent evidence

- Base plus candidate regression test: fails two Qwen3 mapping assertions.
- Exact candidate regression suite: 5 tests pass.
- Independent adversarial checker: all 13 routes resolve to existing cookbook pages, all corresponding legacy source files are absent, the requested popular-model URL targets cookbook intro, Qwen3 targets Qwen3-Next, and all 209 configured redirect sources are unique.
- Redirect generator and Python compilation succeed.
- The candidate changes no runtime or native source. Native rebuild and GPU numerical execution are inapplicable.

## Environment and limitations

The prepared interpreter was `/tmp/amdpilot-repo-j-c54b5ba6f275/venv/bin/python`. Tests ran on x86_64 Linux with Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2.26015, and one visible AMD Instinct MI350X. The GPU was not used because the reviewed behavior is entirely static documentation routing.

The Mintlify CLI was unavailable, so no full Mintlify build or deployed HTTP redirect was tested. This review also does not requalify the semantic accuracy of all model commands migrated by the earlier PR across unavailable model and hardware combinations.
