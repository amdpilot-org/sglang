# Investigation report: Kimi-K3 TP embedding collective divergence

Upstream issue: https://github.com/sgl-project/sglang/issues/37393

Mirror issue: https://github.com/amdpilot-org/sglang/issues/975

## Finding

The prepared `main` checkout already contains the directly relevant correction
from upstream PR https://github.com/sgl-project/sglang/pull/33614, merged on
2026-08-30 after the reported v0.5.18 build commit. The PR identifies rank-local
DSpark sampling and acceptance decisions as a source of TP sequence/KV-state
divergence and synchronizes those decisions before they mutate state.

For the stack in the issue, the important current call site is
`DSparkWorkerV2._forward_prefill`: immediately after obtaining
`batch_output.next_token_ids`, it broadcasts those IDs from rank 0 using
`SpecTpSyncSite.DSPARK_TARGET`, before publishing sequence lengths or entering
the draft path. The v0.5.18 source at commit
`71de97b264b04dcd514cf904003028aefe9775c8` has no `SpecTpSync` import and no
prefill-token broadcast. Thus v0.5.18 can publish a rank-local sample; once TP
ranks commit different tokens, their later chunk sizes can differ and the full
TP `VocabParallelEmbedding` all-reduce is the first collective that exposes the
already-diverged state.

Changing the embedding collective was rejected as a fix. The reported launch
uses DCP rather than DP attention, so Kimi-K3 correctly uses the full TP group
for its sharded vocabulary table. Reducing different row counts cannot be made
valid by selecting a different group; the state divergence must be prevented
at its source.

## Regression coverage

`test/registered/unit/spec/test_spec_tp_sync.py` adds coverage that:

- an 8-rank peer's deliberately divergent prefill sample is overwritten by the
  rank-0 decision, including execution with tensors on the assigned GPU;
- TP=1 is a no-op;
- a deliberately disabled synchronization site is a no-op; and
- the actual DSpark prefill method synchronizes `next_token_ids` before it
  publishes sequence lengths.

The same source assertion fails against the downloaded v0.5.18 implementation
because the `DSPARK_TARGET` synchronization call is absent, and passes against
the prepared checkout.

## Evidence and limitations

Raw logs are retained under
`/tmp/amdpilot-repo-j-e213a23854c4/evidence/`, including the v0.5.18 source,
before/after source comparison, pytest output, and GPU inventory.

The available device was one AMD Instinct MI355X (gfx950), PyTorch
2.11.0+rocm7.2. The GPU test validates the synchronization operation on a real
device tensor, not an 8-rank collective. The Kimi-K3 and DSpark weights and the
reported 8x NVIDIA B300 system were unavailable, so the production workload,
model semantics, NCCL behavior, and long-running concurrency reproduction remain
unverified. The result is therefore `candidate_verified`, not a claim of a full
model reproduction or definitive closure.
