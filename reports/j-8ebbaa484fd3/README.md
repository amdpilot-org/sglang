# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/33656

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1782

The prepared base still contained the follower-rank completion bug described in
the upstream issue discussion. `HybridCacheController._page_backup` skips the
replicated MLA/DeepSeek-V4 writes on nonzero TP ranks, as intended, but initialized
the remaining sidecar result from `bool(backup_transfers)`. A follower with no
rank-sharded sidecars therefore reported zero completed tokens even though there
was no local work to fail and TP0 owned all replicated writes.

The focused regression failed before the source change with `0 == 128`, while
the independent successful-sidecar and failed-sidecar cases passed. After the
change, all three pass. Adjacent HiCache assembly and Mooncake grouping tests
also pass.

This is a candidate-level verification of the issue-specific accounting defect,
not a claim of full production reproduction. The available host has one AMD
gfx950 using ROCm 7.2, not the reported 8x NVIDIA H20 CUDA topology, and the
DeepSeek-V4-Flash-0731 weights and workload were unavailable. Consequently no
TAIL_K_SWA canary, logits, NaN sampling, semantic accuracy, or distributed
Mooncake claim is made.

Raw outputs are retained under
`/tmp/amdpilot-repo-j-8ebbaa484fd3/evidence/`.
