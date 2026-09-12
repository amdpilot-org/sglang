# MiniMax-M3 top-level composition investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/32286

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2103

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The prepared checkout already contains the source fix. Upstream PR
https://github.com/sgl-project/sglang/pull/36626 merged on 2026-08-28 and changed
`MinimaxM3Detector._get_child_schema()` to use the shared recursive
`get_schema_properties()` helper. That helper collects properties beneath
top-level `anyOf`, `oneOf`, and `allOf` branches.

The existing test covered top-level `oneOf` scalar coercion only. This change
adds the issue's exact streaming shape: three calls whose arguments are a
number, an array of strings, and an array of numbers. It also exercises
top-level `anyOf` and `allOf` independently.

## Failing-before evidence

`raw/legacy_lookup_regression.txt` runs the new regression after replacing the
helper at runtime with the former implementation's effective lookup:

```python
schema.get("properties", {})
```

The output reproduces the issue exactly: `number` is `"42"`, while both arrays
contain raw `]<]minimax[>[<item>...` tags. The independent `anyOf` and `allOf`
cases also return `"7"` instead of `7`. The command exits 1.

## Passing-current evidence

`raw/fixed_minimax_m3_pytest.txt` records the focused suite on the actual
checkout: 26 tests and 16 subtests pass, including the exact streaming case and
the independent composition boundaries.

`raw/baseline_minimax_m3_pytest.txt` records the unchanged checkout before the
new tests: 24 tests and 14 subtests pass.

`raw/gpu_inventory.txt` records the assigned single gfx950 device. GPU execution
was not relevant to this deterministic parser path and was not performed.

## Limitations

MiniMax-M3 weights were not present, and the reported server command requires
four GPUs while the job assigned one. Consequently, this is not a full-model,
HTTP, semantic-accuracy, or distributed-workload reproduction. The tiny Llama
fixture cannot validate MiniMax-M3-specific model output, so it was not used.
