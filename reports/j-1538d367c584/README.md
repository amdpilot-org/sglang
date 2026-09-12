# Independent review of PR 2906 at `7f6725c124ac107fbc06daa0810757488cb6ea28`

Upstream issue: https://github.com/sgl-project/sglang/issues/19090

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2940

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2906

Recommendation: **request changes**. The candidate is a partial local API improvement, not a full resolution of the original issue.

The recorded base already contains the diffusion memory controller, scheduler handlers, and HTTP sleep/wake routes. The candidate adds `DiffGenerator` methods, documentation, mocked request tests, and error-payload handling. Its stated transport-exception normalization is incomplete: a forwarding `ConnectionError` escapes unchanged. More importantly, it does not provide the original contract's real diffusion-model sleep/wake/refit validation or comparison against kill-and-relaunch, and FSDP remains unsupported.

## Evidence

- Base `358c163250ad3b1f62939b01ce1314a0a31a0365`: the candidate's three `DiffGenerator` methods are absent (exit 1).
- Candidate exact commit: its focused suite passes, 11 tests total.
- Candidate exact commit: an independent transport exception case fails because `ConnectionError` is not normalized.
- Candidate exact commit: a real GPU controller round trip passes on AMD Instinct MI355X/gfx950 using Torch 2.11.0+rocm7.2 and HIP 7.2.26015. It releases 4,198,400 allocated bytes for the test module and returns output within `8.344650268554688e-07` max absolute error of a CPU float64-derived reference.
- Imports resolved to the checked-out source under `/job/repo/python/sglang/...`. No native files changed, so a native rebuild was not applicable.

Raw command output is retained in `raw/`. Full model and benchmark claims remain unverified because no diffusion checkpoint was available. The deterministic tiny Llama fixture was intentionally not used as proof for a diffusion architecture.
