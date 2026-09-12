# Independent review of PR 1542

Reviewed exact candidate `3df72284c7a4fbfb906582ac86a2628dedc33788` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. This is a partial original-issue fix.

The candidate successfully repairs the previously reported DP-attention direct-linker gap. Its registered configuration path now propagates `get_parallel().attn_dp_rank`; the resulting embedded Mooncake clients select distinct `rank_0_0_0` and `rank_1_0_0` directories in the independent fixture. It also adds the requested documentation that Mooncake LOCAL_DISK supports one client per final directory. The candidate regression passed, as did fifteen adjacent Mooncake tests.

It does not fully establish the documented guarantee that every embedded client receives a private directory:

1. Pure-DP replicas have independent caches but `attn_dp_rank == 0` for every replica by design. The candidate has no pure-DP replica identity available in `_direct_linker_storage_config`, and two such clients both selected `rank_0_0_0`.
2. Attention context-parallel workers have distinct `attn_cp_rank` values, but the final directory name excludes that rank. Two CP workers likewise both selected `rank_0_0_0`.

These are path-selection counterexamples to the same Mooncake one-client-per-directory contract, not unrelated serving smoke failures. Raw commands, fixtures, outputs, the candidate diff, and environment evidence are retained under `raw/`.

No native source changed, so no native rebuild was applicable. Imports resolved from the checked-out source under `/job/repo/python`. One AMD Instinct MI355X (gfx950) was visible with torch 2.11.0+rocm7.2/HIP 7.2, but this deterministic Python rank/path review did not execute GPU kernels. DeepSeek-V4-Flash weights, a TP8/multi-node deployment, and a live Mooncake LOCAL_DISK cluster were unavailable, so no full-model corruption or restart-recovery qualification is claimed.

Upstream issue: https://github.com/sgl-project/sglang/issues/35484

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1574

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1542
