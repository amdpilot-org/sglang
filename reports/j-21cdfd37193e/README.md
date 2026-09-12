# Investigation report: DFlash2 TP greedy divergence

Upstream issue: https://github.com/sgl-project/sglang/issues/38009

Mirror issue: https://github.com/amdpilot-org/sglang/issues/826

The selected source already contains a narrowly matching correction. The
reported image identifies commit `5f55db35e926d50676f75b812640ea2410b0fe0e`
(2026-08-22). Upstream PR #33614 merged eight days later as commit
`f60bc73c5836d45457575c17f4722c6bd60f06b0`, and that commit is an ancestor of
the prepared base. It broadcasts DFlash target predictions and acceptance
decisions from TP rank 0 before accepted tokens and recurrent/KV state are
committed. This directly addresses rank-local greedy decisions causing TP
state divergence.

The added parameterized regression checks all acceptance boundaries:

- greedy verification synchronizes `DFLASH_ACCEPT_GREEDY`;
- stochastic verification synchronizes both accept length and bonus token;
- selector verification synchronizes both accept length and bonus token.

The full DFlash logits unit file passes (13 tests). Loading the exact reported
commit from an isolated worktree and checking for the same synchronized
acceptance boundary fails: that implementation has no `_accept_block` method.
Raw logs are retained under `/tmp/amdpilot-repo-j-21cdfd37193e/` as
`test-dflash-logits.log`, `current-regression.log`,
`accept-path-regression.log`, `pre-fix-regression.log`, and
`gpu-evidence.log`.

This is not a full reproduction of the issue. The assigned machine exposes one
AMD Instinct MI350X/gfx950, while the report used four RTX 3090 GPUs at TP=4,
and neither 27B checkpoint is present. A real ROCm tensor operation was run to
identify and exercise the assigned GPU, but no claim is made that it validates
the Qwen3.8 architecture, CUDA/NCCL behavior, multi-rank execution, or the
reported token sequence.
