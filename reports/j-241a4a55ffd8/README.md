# Independent review of PR 2806

Reviewed exact candidate commit `2b9bcb31501441d2cc36f3a1a3faac04084d3689` against upstream issue https://github.com/sgl-project/sglang/issues/32750 and candidate mirror issue https://github.com/amdpilot-org/sglang/issues/2739, using recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

The base cannot collect the candidate's PP-context regression because the partial-projection implementation is absent (`ImportError: _BlockFp8LinearSlice`). On the exact candidate, the focused suite passed (69 tests and 12 subtests), and an independent GPU reference confirmed that noncontiguous/out-of-order stage-local projection contributions summed before one RMSNorm match the full mathematical reference exactly (`max_abs=0.0`). Invalid widths, empty feature lists, and out-of-range indices were rejected.

The candidate changes Python only. The imported `sglang` source was `/job/repo/python/sglang/__init__.py`; therefore no native rebuild was applicable. Execution used torch `2.11.0+rocm7.2`, HIP `7.2.26015`, and one AMD Instinct MI350X.

Recommendation is `unverified`, not rejection: the implementation and protocol-level tests support the proposed design, but the available single-GPU environment cannot execute real multi-rank PP + PD, and Kimi-K3 / DeepSeek-V4 DSpark weights were unavailable. Actual Mooncake/NIXL transfer with final-stage draft-KV ownership and real quantized checkpoint execution remain counterexamples that were not executable here.
