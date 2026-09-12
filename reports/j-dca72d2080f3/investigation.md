# DeepSeek-V4-Pro padded TP shard investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32781

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1988

## Finding

The recorded base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) already contains the
issue-specific correction in `FusedMoE._load_w13` and `FusedMoE._load_w2`.
Both methods derive the source shard width from the unpadded checkpoint tensor,
use that width for the TP offset, and copy the source shard into the leading
part of the padded destination while leaving its tail zeroed.

The missing piece was direct regression coverage. The added tests use the
reported DeepSeek-V4-Pro dimensions and high ranks:

- w13: checkpoint width 3072, TP16 source shard 192, destination shard 256,
  including ranks 13 and 15;
- w2 MXFP4 scale: checkpoint width 96, TP16 source shard 6, destination shard
  8, including ranks 13, 14, and 15;
- an unpadded destination boundary and split w3 logical-half placement.

Temporarily substituting the reported old calculation
`shard_size * tp_rank` for the recorded-base implementation produced seven
failures. Rank 15 reproduced `IndexError: ... got 3840` in `_load_w13`.
Restoring the recorded-base implementation made all eight tests pass.

## Hardware scope

The assigned GPU is one AMD Instinct MI355X (gfx950), not an NVIDIA H100. The
actual `_load_w13` and `_load_w2` methods were executed on that GPU and matched
independent tensor references: w13 rank 15 selected 2880 through 3071 and left
64 zeros; w2 scale rank 14 selected 84 through 89 and left 2 zeros.

This does not validate the CUDA-only Marlin kernel, the DeepSeek-V4-Pro model
architecture or weights, a TP16 process group, or the reported two-node H100
deployment. No model weights were available, and no full serving claim is
made.

## Raw evidence

Raw command output is retained outside the worktree in the prepared private
runtime directory:

- `/tmp/amdpilot-repo-j-dca72d2080f3/evidence/padded_tp_weight_loading_old_logic.txt`
- `/tmp/amdpilot-repo-j-dca72d2080f3/evidence/padded_tp_weight_loading_cpu.txt`
- `/tmp/amdpilot-repo-j-dca72d2080f3/evidence/padded_tp_weight_loading_gfx950.txt`
