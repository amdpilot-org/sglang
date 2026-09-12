# Independent review of PR 1980

Upstream issue: https://github.com/sgl-project/sglang/issues/33292

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2014

Candidate: https://github.com/amdpilot-org/sglang/pull/1980 at `37ae11682644b4be75ad81ed7224ed38b31557c4`

Recommendation: **accept**. The candidate fully corrects the source-level contract described by the issue. On the recorded base, selecting XPU while constructing a subclass that only implements `forward_native` raises `AttributeError`. At the exact candidate commit, the new base `forward_xpu` delegates to `forward_native`; the candidate regression and independent boundary cases pass.

The review imported `custom_op.py` from `/job/repo/python/sglang/multimodal_gen/runtime/layers/custom_op.py`, not from an installed copy. The candidate contains no native-code changes, so no native rebuild applies.

This host has an AMD Instinct MI355X/gfx950 and ROCm PyTorch, not Intel XPU hardware or an XPU-enabled PyTorch build. Platform selection was patched at the dispatch boundary. Therefore actual Intel device execution, full multimodal model loading, semantic model accuracy, and distributed behavior were not tested or claimed. Those limitations do not expose a remaining counterexample to the reported missing-method contract.

Raw outputs are retained in `raw/`. The failing base output records the construction-time traceback; candidate outputs record the three regression passes and independent adversarial passes.
