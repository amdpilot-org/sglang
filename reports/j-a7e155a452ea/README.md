# Investigation report: sglang#31384

Upstream issue: https://github.com/sgl-project/sglang/issues/31384

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2256

## Finding

The checked-out implementation reproduced the reported metadata-shape defect at
the shared DSA padding calculation. `ForwardBatch.prepare_mlp_sync_batch` rounds
each DP rank's token count up to `attn_tp_size`, but `cal_padded_tokens` selected
the raw counts. For `[11, 1]` and `attn_tp_size=8`, attention metadata was sized
for 11/1 rows while the forward batch was padded to 16/8 rows. This is the same
query/KV batch disagreement that FA3 reports as `batch_size must be equal to
batch_size_k`.

The fix aligns every DP rank count before applying either MAX_LEN or SUM_LEN
selection, matching the production padding order. It is deliberately confined
to the DSA metadata helper.

## Related changes inspected

- sgl-project/sglang#29262 describes stale EAGLE attention metadata after DP
  padding, but its generic re-plan approach is not safe for the current DSA
  multi-step wrapper.
- sgl-project/sglang#30642 identifies the same GLM-5.x DSA speculative failure
  class and independently proposes aligning `cal_padded_tokens` to
  `attn_tp_size`.
- sgl-project/sglang#26016 covers additional DP-attention/EAGLE configurations
  and FlashInfer draft-extend padding. Those broader changes are outside this
  FA3/DSA correction.

## Evidence

- `raw/regression-before.txt`: the added regression against the original base;
  unaligned MAX_LEN and SUM_LEN cases fail with `11 != 16`.
- `raw/regression-after.txt`: the regression and existing forward-metadata plan
  tests pass after the correction (12 tests plus 2 subtests).
- `raw/gpu-padding-check.txt`: on the assigned AMD Instinct MI355X (`gfx950`),
  actual GPU tensors are padded to 16/16 rows for MAX_LEN and 16/8 for SUM_LEN,
  preserving real rows and zero-filling the tail.

## Limitations

The reported deployment requires two H20 nodes, TP16/EP16/DP2, GLM-5.2-FP8
weights, CUDA FA3, DeepEP, and concurrent serving traffic. This environment has
one gfx950 GPU and no GLM-5.2 weights, so that complete model/distributed
reproduction and semantic-accuracy validation remain unverified. The GPU test
qualifies the corrected metadata sizing on ROCm only; it does not substitute for
the missing CUDA multi-node workload. No native library was changed or rebuilt.
