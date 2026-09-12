# Investigation: DSpark concurrency=1 launch failure

Upstream issue: https://github.com/sgl-project/sglang/issues/34522

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1582

## Finding

The prepared `main` source already contains the accepted fix. No production
code change is justified.

The upstream issue discussion identifies PR #32954 as the fix, and the
reporter explicitly confirmed: “Yeah, #32954 fixes it.” That PR merged as
`166c6f71811087d427feed48c685356965810707`, after the v0.5.17 release commit
`29481685462732237d80d86076d6563e1f658102` used by the report.

`git blame` on the prepared source attributes the trailing two-CTA barrier at
`python/sglang/kernels/ops/gemm/cutedsl_bf16_gemm.py:496` to that same merge
commit. The barrier follows all warp dispatch branches and is guarded by
`cutlass.const_expr(self.use_2cta)`, so one-CTA tactics are unchanged.

The current selection code independently supports the concurrency-specific
diagnosis: Kimi-K3 shapes use TGV for `m <= 8`, and the tactic ladder begins
with tactic 18, a two-CTA tactic. The original failure was asynchronous and
therefore surfaced later at the scheduler stream synchronization.

## Regression evidence

`test_tgv_exit_barrier.py` parses the kernel without importing NVIDIA-only
CuTe DSL dependencies. It requires a gated `cluster_arrive_relaxed()` plus
`cluster_wait()` after every warp dispatch branch.

- Current prepared source: 3 tests pass.
- Exact v0.5.17 source: the checker fails with `missing two-CTA cluster
  arrive/wait exit barrier`.
- Boundary cases: the checker rejects removal of the wait and rejects moving
  the barrier before warp dispatch.

Raw outputs and the extracted v0.5.17 source are retained under
`/tmp/amdpilot-repo-j-0c4ad14625fe/evidence/`.

## Hardware limitation

The assigned device is one AMD Instinct MI355X (`gfx950`) with ROCm 7.2. The
affected CuTe DSL TGV kernel requires NVIDIA SM10x, while the reported setup
used eight NVIDIA L20D/SM103 GPUs and Kimi-K3 plus its DSpark draft weights.
Those weights and that architecture were unavailable. Consequently this work
does not claim a full serving, model-semantic, CUDA-graph, or multi-node
reproduction. A small gfx950 GEMM was checked against a CPU float64 reference
only to verify real execution on the assigned GPU.
