# Independent review of PR 997

Candidate: https://github.com/amdpilot-org/sglang/pull/997 at exact commit
`7035931a1eee025b1847b58c531c583f983ea8a6`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38167

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1818

## Recommendation

**Request changes.** The candidate is a useful partial hardening of the
monolithic scheduler: it recognizes the two reported fatal strings, returns a
restart-required error, retains the RPC loop, and rejects later monolithic
requests without worker dispatch. Its focused 34-test suite and its regression
probe pass at the exact commit.

It does not fully resolve the original issue. An independent probe at the exact
tip confirms that `scheduler_rpc_timeout=None` leaves ZMQ `RCVTIMEO=-1`, and a
disaggregated encoder event loop catches the same fatal signature and dispatches
another work item. The change also does not turn the initial 1344x768 failure
into an admission or insufficient-memory error. The candidate's string matching
cannot establish that `device not ready` was caused by memory exhaustion.

## Reproduction and evidence

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, a direct behavioral
probe confirms there is no fatal-state latch and a later request reaches its
worker handler (`raw/base_behavior.txt`). On candidate `7035931...`, the
candidate regression probe passes and the full focused command reports 34
passing tests (`raw/candidate_regression_probe.txt` and
`raw/candidate_focused_tests.txt`).

The independent adversarial probe (`raw/candidate_adversarial.txt`) measured:

- grouped monolithic work is rejected after the fatal latch without dispatch;
- the default RPC receive timeout resolves to `None` and the actual ZMQ socket
  retains `RCVTIMEO=-1`;
- a disaggregated encoder loop executes a second work item after a first item
  raises `CUDA driver error: device not ready`, with no fatal state latched.

## Environment and limitations

Imports resolved to the source checkout, including
`/job/repo/python/sglang/multimodal_gen/runtime/managers/scheduler.py`. The
prepared interpreter has Torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and one AMD
Instinct MI350X (`gfx950` capability). A float64 GPU sum-of-squares matched the
independent integer reference; this establishes device execution only.

The reported RTX 5070 12 GB / WSL2 / CUDA 13.0 environment, MiniMax-H3 weights,
and LoRA were unavailable. Therefore the 1344x768 failure, its resource cause,
and the post-failure 480p request were not reproduced. The tiny Llama fixture
from PR 649 was inspected but is an autoregressive transport/engine fixture and
cannot qualify the MiniMax-H3 diffusion architecture or this CUDA allocator
failure, so it was not substituted as proof. No native files changed and no
native rebuild was required.
