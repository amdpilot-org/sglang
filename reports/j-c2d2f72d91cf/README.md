# Independent review of PR 2107

Candidate: https://github.com/amdpilot-org/sglang/pull/2107 at
`28dcbbf30d215f1a487c671de86753d043621860`

Upstream issue: https://github.com/sgl-project/sglang/issues/33483

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2143

## Recommendation

Accept. The candidate is a behavioral correction, not warning-only or
test-only hardening. On the recorded base, the issue inputs
`token_capacity=3145118`, `context_len=32768`, and decode graph maximum 32
resolve to 4096 running requests. At the exact candidate commit, implicit
admission is capped to 32. This prevents the default scheduler admission limit
from allowing the running batch to cross the largest captured decode graph
shape, eliminating the source-level precondition for the reported absorbing
eager-fallback state.

Explicit `--max-running-requests` remains authoritative and warns once when its
per-worker value exceeds graph coverage. Disabled graphs and configurations
without a graph maximum retain capacity-derived admission.

## Independent evidence

- The exact candidate regression passed at the candidate commit: 6 passed.
- The same test file run from outside the checkout on recorded base failed in
  the two expected behavioral locations: the issue default was 4096 rather
  than 32, and the candidate-only warning helper did not exist. The other four
  cases passed.
- An independent nine-case boundary suite passed on the candidate, covering
  graph maxima 32, 128, and 8192; a 40-token pool; disabled and absent graph
  configurations; explicit overrides; and DP-local explicit limits.
- The independent suite plus the adjacent pool configurator suite passed: 50
  tests and 7 subtests.
- Imports resolved to the checked-out source under
  `/job/repo/python/sglang`, not a separately installed SGLang package.
- Candidate diff whitespace validation passed.

The first run of the independent suite used an incomplete lightweight test
object and failed after calculation because the warning method was not bound.
That harness error was corrected and the full suite rerun; it is not counted as
a product failure.

## Architecture and environment limits

The prepared interpreter is PyTorch `2.11.0+rocm7.2` on one AMD Instinct
MI350X (`gfx950`), not the reported NVIDIA L40. The Qwen2.5-0.5B weights and
ShareGPT workload were not available, so the reported stochastic 4x latency
transition was not rerun. GPU execution was not used as review evidence. This
review verifies the deterministic admission/graph-capacity contract and its
boundaries, not L40 performance magnitude or queueing latency. No native source
changed, so no native rebuild was required or performed.
