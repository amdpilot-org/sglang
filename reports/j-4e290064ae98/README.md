# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/37585

Mirror issue: https://github.com/amdpilot-org/sglang/issues/944

Base `358c163250ad3b1f62939b01ce1314a0a31a0365` still used TP-only padding in `ForwardBatch.prepare_mlp_sync_batch`. Running the focused test from upstream PR #37587 unchanged against this checkout failed at the reported boundaries: B=38,D=6,A=8 produced 232 physical rows, and B=33 produced 200. Both are divisible by A but not D.

The patch applies the shared correction from the inspected open upstream PR at its current head `27a790d6b285d189306a6aaaaa211c6f969866a0`: fixed-width hybrid target verification and synchronized idle ranks align to `lcm(A, D)` before DP padding selection. It also preserves the accompanying dummy-request contract: dummy rows do not read nonexistent state/tree metadata and their physical outputs are zero-initialized before downstream consumption.

After the change, the layout suite passes 5 tests and 7 boundary subtests. The GPU suite passes 17 cases on the assigned AMD Instinct MI355X gfx950, comparing real rows with independently executed unpadded references and checking dummy outputs and persistent state exactly.

Raw logs are retained in `raw/`. The full Qwen TP8 H20 serving reproduction remains unavailable because this job has one AMD GPU and no Qwen3.5-35B-A3B weights. No claim is made for multi-rank collectives, CUDA/H20 execution, semantic accuracy, or the separate GLM DSA KPool boundary.
