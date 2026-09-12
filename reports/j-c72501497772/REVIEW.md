# Independent review of PR 768 at `ca540bb`

Recommendation: **accept**. The exact candidate fully fixes the original shared-pool break-input lifetime contract on the available ROCm architecture.

The prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) still used weak references in the actual breakable graph bridge. On one MI355X, three independent shapes all reused the bridge input address in a later capture. Replaying the earlier graph then changed the later bucket's live buffer from `-999` to `11`. The standalone allocator control also showed both directions of corruption with weak references and neither with strong references.

At exact candidate `ca540bbde8b079c53aa63885931b6d2c950c5127`, the same actual-bridge test allocated distinct addresses for every shape, returned deterministic `22` outputs, and left the later buffers at `-999`. The candidate's focused regression passed, and the complete `TestBreakableCUDAGraphBasic` class passed (10 tests plus two subtests).

The candidate is a source fix plus regression hardening, not test-only hardening: it removes weak conversion of captured break arguments and holds the original argument/keyword objects strongly in each replay closure. No native files changed. The interpreter imported SGLang and the bridge module from `/job/repo/python`, so the checked-out source was exercised rather than an installed SGLang wheel.

Raw review evidence was preserved outside revision switches under `/job/review-evidence-j-c72501497772/`, including the exact candidate diff, issue/PR JSON, independent scripts, and base/candidate logs.

## Limits

The machine provides one AMD Instinct MI355X (`gfx950`), Torch `2.11.0+rocm7.2`, and HIP `7.2.26015`. It does not provide the reported B300/CUDA TP8 environment or the Hy4/GLM/Kimi weights. Therefore this review verifies the root allocator/bridge contract but does not independently establish the large-model greedy-token, illegal-memory-access, semantic-accuracy, capture-memory, or throughput claims. There was no native change and the prepared environment declares no native artifact, so a native rebuild was not applicable.

Upstream issue: https://github.com/sgl-project/sglang/issues/37606

Mirror issue: https://github.com/amdpilot-org/sglang/issues/792
