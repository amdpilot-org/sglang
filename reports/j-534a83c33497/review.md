# Independent review of amdpilot-org/sglang PR 2824

Candidate reviewed: `f3ba3f6883a434ebcc0f9841845f37156bc7d5eb`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Upstream issue: https://github.com/sgl-project/sglang/issues/32124

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2752

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2858

## Recommendation

Request changes. The candidate implements and tests an opt-in, atomic marker and
does eliminate the repeated mocked all-M dispatch when the marker matches, but it
does not fully implement the requested exact DeepGEMM/JIT identity.

## Evidence and finding

On the exact base, an independent process-equivalent reproduction cleared the
in-memory initialization state between calls while retaining the cache and saw
two warmup calls. On the exact candidate, the same contract with marker reuse saw
one warmup call and one marker. The candidate's nine regression tests also pass.

The blocking counterexample is `_module_identity`: it hashes only
`deep_gemm.__file__`, normally the package `__init__.py`, and the distribution
version. An independently executed adversarial test changed a sibling
`jit_kernels.py` generator while keeping `__init__.py` and the version fixed.
The candidate returned the same identity before and after
(`identity_changed: False`). A restored cache can therefore be accepted after a
locally patched/rebuilt JIT implementation changes, contrary to the issue's
requirement that the marker identify DeepGEMM/JIT exactly. The marker needs a
reliable build/source identity that covers the native/JIT implementation (for
example an authoritative DeepGEMM build revision or comprehensive installed
artifact manifest), with unknown identity handled conservatively.

This is a source-level, independently reproduced partial fix. It is not merely
test-only hardening, because the candidate adds working marker behavior. It is
not a full original-issue fix because the compatibility key has a demonstrated
false match.

## Architecture and environment limitations

The prepared interpreter is Python 3.12.3 with Torch `2.11.0+rocm7.2`, HIP
`7.2.26015`, and one AMD Instinct MI350X. `torch.version.cuda` is `None` and
`deep_gemm` is not installed. SGLang and the candidate marker module imported
from `/job/repo/python`, while Torch imported from
`/opt/venv/lib/python3.12/site-packages/torch` through the prepared interpreter.

No NVIDIA GPU, CUDA DeepGEMM execution, DeepSeek-V4-Pro weights, or TP=8 hardware
was available. Consequently real JIT compilation/cache restoration, independent
GPU numerical parity, CUDA graph parity, steady-state performance, and startup
latency remain unverified. The candidate changes Python only; no native source
changed, so a native rebuild was not applicable.

Raw command output and the independent scripts are preserved outside the
checkout in `/job/review-evidence-j-534a83c33497/` so revision switches could not
overwrite them.
