# Independent review of PR 723

Candidate: https://github.com/amdpilot-org/sglang/pull/723 at `f2e1f3240e87f0cb0baf46fbef533db34c434a71`

Upstream issue: https://github.com/sgl-project/sglang/issues/38981

Mirror issue: https://github.com/amdpilot-org/sglang/issues/727

Recommendation: **request changes**.

The prepared base reproduced the reported failure through the real
`free_kv_row_segments` implementation: touching slices `[0,5)` and `[5,7)`
with page size 4 raised `AssertionError: segment at 5 shares a page with the
one ending at 5`. The exact candidate coalesced those slices successfully and
all five of its focused regressions passed.

The retry mechanism is not generally idempotent. It stores only the most
recent fingerprint on the shared allocator. In an independent `A, B, A`
sequence, A initially released two pages, B released one, and retrying A
released A's same two pages again. The expected increments were `2, 1, 0`;
the observed increments were `2, 1, 2`. This was reproduced with CPU tensors
and on the assigned AMD Instinct MI350X.

Thus the coalescing portion fixes the original shared-boundary assertion, but
the submitted consolidation does not satisfy the approved requirement that
cleanup remain safe and idempotent across interleaved segments and retries.
The candidate's immediate-repeat tests do not cover this counterexample.

The production NVIDIA B300/SM103, eight-rank Kimi-K3 configuration was not
available. The prepared machine has one AMD Instinct MI350X (gfx950), ROCm
7.2, and no model weights. The change is Python-only; imports were confirmed
from the candidate checkout at
`/job/repo/python/sglang/srt/mem_cache/common.py`, and no native rebuild was
applicable (`native` and `wheel` are null in the prepared environment).

Raw logs and the standalone probe were preserved outside the checkout under
`/job/evidence-j-836f53935809/` while revisions were switched.
