# Independent review of PR 1693

Reviewed exact commit `8e8b1b27682a672a5252ad0f7ac4f63cac8be097`
against upstream issue https://github.com/sgl-project/sglang/issues/35484 and
mirror issue https://github.com/amdpilot-org/sglang/issues/1731.

Recommendation: accept.

The recorded base reproduces the original path-sharing defect and both concrete
counterexamples from the prior review. Pure-DP replicas and distinct attention
context-parallel workers select the same configured directory. An independent
Cartesian DP/TP/PP/CP probe found only one unique final path for 16 logical
clients.

At the exact candidate commit, the candidate's focused regression passes and the
same independent probe finds 16 unique final directories. Pure-DP replicas are
separated by scheduler DP rank, attention-CP workers are separated by the
conditional `_cp<rank>` suffix, and TP/PP remain part of the directory identity.
Inspection of the actual construction path confirms `ps.dp_rank` is copied into
`CacheInitParams` and the registered `MooncakeDirectLinker` copies that value into
`HiCacheStorageConfig`; this avoids relying on `attn_dp_rank`, which is zero for
pure DP.

This is a full source-level fix for the original SGLang directory-isolation
contract, not merely test hardening. It does not prove Mooncake runtime behavior
outside that contract: no external Mooncake service, DeepSeek-V4-Flash weights,
TP8/multi-node deployment, live bucket collision, or restart-recovery workload
was available. The Mooncake-side recovery change remains outside this repository.
One AMD Instinct MI350X was visible, but GPU execution is irrelevant to the
deterministic Python rank propagation and path selection under review and was not
claimed. No native files changed, so no native rebuild was applicable.
