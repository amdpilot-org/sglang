# Investigation: unified DeepSeek V4 prefill CP arguments

Upstream issue: https://github.com/sgl-project/sglang/issues/32553

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2033

## Finding

The reported v0.5.16 argument-validation defect is already fixed at the prepared
base (`358c163250ad3b1f62939b01ce1314a0a31a0365`). No runtime source correction
was justified. This change adds a focused regression for the canonical
`--enable-prefill-cp --cp-strategy interleave` path and two independent boundary
cases.

The v0.5.16 sources retained both canonical and legacy CP fields. Its startup
sequence called `_handle_legacy_cp_arguments`, then the DeepSeek V4 hook, then
`_handle_legacy_cp_arguments` again. The first projection could set
`enable_prefill_context_parallel`; `validate_deepseek_v4_cp` then unconditionally
set `enable_dsa_prefill_context_parallel`. `_handle_context_parallelism` rejected
that resulting pair as mutually exclusive. Exact tag sources are retained in
`raw/server_args-v0.5.16.py` and `raw/deepseek_v4_hook-v0.5.16.py`.

Current source has only the canonical `enable_prefill_cp` and `cp_strategy`
fields. The DeepSeek V4 hook declares resolved values for DP attention,
`moe_dense_tp_size`, and `attn_cp_size`; it no longer writes either legacy
enable flag. The regression parses the reported canonical CLI pair, applies the
actual DeepSeek V4 validator and general CP handler under a CUDA platform
override, and verifies interleave initialization without legacy fields.

Related upstream changes inspected:

- https://github.com/sgl-project/sglang/pull/27312 introduced the unified CLI
  while retaining compatibility projection to legacy fields.
- https://github.com/sgl-project/sglang/pull/38293 removed legacy prefill-CP
  fields/runtime paths and intentionally deprecated prefill CP on HIP/NPU/MUSA.
- https://github.com/sgl-project/sglang/pull/36229 canonicalized the remaining
  strategy-based CP APIs.

## Limitations

The reported workload needs DeepSeek-V4-Flash-FP8 weights and eight NVIDIA H20
GPUs. Neither is available. The assigned device is one AMD Instinct MI355X
(gfx950), and current source intentionally rejects prefill CP on HIP before
model lookup. Therefore no full model startup, eight-rank execution, CUDA/H20
kernel execution, semantic accuracy, or distributed performance claim is made.
The added test qualifies the reported argument-validation conflict only.

The broader `TestContextParallelServerArgs` class has two existing tests without
a CUDA platform override; on this HIP host they hit the intentional platform
deprecation guard. That raw nonzero run is retained as
`raw/context-parallel-class.txt` and was not treated as an issue-specific
regression.
