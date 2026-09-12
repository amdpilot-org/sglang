# Independent review of PR 2946

Candidate: https://github.com/amdpilot-org/sglang/pull/2946 at `e9952a5825d098de1c068d061062fe2522981d27`

Upstream issue: https://github.com/sgl-project/sglang/issues/32485

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2982

## Recommendation

Request changes. The candidate is a partial fix: it implements pre-stream shard
selection for the two named layouts and fixes the earlier unindexed
`mtp.safetensors` regression, but it does not satisfy the original contract's
safe fallback when an index mixes a recognized requested-layer entry with an
unknown draft layout.

## Independent findings

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the
HF-layout regression fails because all five prepared shards reach the RunAI
iterator instead of only the requested-layer and shared shards.

At the earlier candidate `8c799873170db6688e35734e591e36a5f188eaa5`, the
reported incomplete-index counterexample reproduces: after
`maybe_add_mtp_safetensors` appends `mtp.safetensors`, the iterator receives
only `draft-1.safetensors`.

At the reviewed candidate `e9952a5825d098de1c068d061062fe2522981d27`, all 16
candidate tests pass and the earlier auto-added-file counterexample is fixed.
However, an index containing all prepared shard filenames and these keys still
violates the safe fallback requirement:

```text
model.layers.0.weight                  -> target.safetensors
mtp.1.decoder.weight                   -> draft-requested.safetensors
model.nextn.layers.1.shared.weight     -> draft-unknown.safetensors
```

The recognized `mtp.1.*` key sets `requested_layer_found`, after which the
selector returns only `draft-requested.safetensors`. It silently drops the
unknown-layout draft shard rather than returning the complete prepared list.
The candidate's unknown-layout test does not expose this because it tests an
index with no recognized requested-layer entry, which takes a different
fallback branch.

## Environment and scope

The interpreter imported `sglang.srt.model_loader.loader` from
`/job/repo/python/sglang/srt/model_loader/loader.py` on both base and candidate.
No native files changed, so no native rebuild was applicable. The behavior is
deterministic Python JSON/path selection before tensor streaming; GPU numerical
execution would not measure this contract and was not performed. No live object
storage credentials or production MTP checkpoint were available, so remote I/O
reduction, distributed loading, and end-to-end model semantics remain
unverified.

Raw command outputs are included alongside this report.
