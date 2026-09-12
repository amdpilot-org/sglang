# Investigation report

The prepared `main` checkout already contains the production solution for the
failure reported in https://github.com/sgl-project/sglang/issues/26794 (mirror:
https://github.com/amdpilot-org/sglang/issues/2476).

The failure depended on NPU post-processing transposing the persistent
`w13_weight` and `w2_weight` parameters into the grouped-matmul kernel layout.
A later disk reload then treated those parameters as canonical load-time
buffers, producing the reported shape error for separate `gate_proj` and
`up_proj` checkpoint tensors. Merged changes #26717 and #29503 first restored
the canonical layout before reload and then removed the persistent transpose
entirely: current code keeps canonical parameter shapes and transposes at the
NPU grouped-matmul call site.

This change adds the missing deterministic regression. It invokes the actual
NPU unquantized post-processing branch with only the unavailable format-cast
operation mocked, then reloads separate gate, up, and down projections through
the actual `FusedMoE` loader. It covers the smallest non-empty intermediate
dimension as a boundary and separately verifies that both halves replace old
values. Raw test, pre-fix-layout failure, and gfx950 numerical outputs are in
`raw/`.

An Ascend device and the reported DeepSeekV3.2 checkpoint were unavailable.
Consequently this report does not claim a full HTTP serving reproduction or
NPU kernel validation. The gfx950 run only validates the tensor loader with the
reported 1408-row expert dimension.
