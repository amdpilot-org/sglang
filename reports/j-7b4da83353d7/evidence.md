# Qwen 3.8 Flash Next PLE scale investigation

## Source and history

- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Source path: `python/sglang/srt/models/qwen4_exp.py`
- The Qwen 3.8 implementation was merged upstream in PR 37500 at commit
  `52fecfdf0908dca24f4c6799ff5967125cc4110e` on 2026-09-08, after the issue's
  2026-08-27 snapshot.
- Its loader routes PLE persistent buffers through
  `_load_qwen4_exp_ple_buffer` before the generic assertion for unknown,
  non-unit scale tensors. `weight_scale` is explicitly included in that route.

## Reported-value reproduction

The regression uses the tensor name and scalar from the report:

```text
model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale
0.00019931793212890625
```

Removing only `"weight_scale"` from the existing PLE buffer allowlist makes the
reported-value regression fail with exit code 1 at the original assertion:

```text
FAILED test/registered/unit/models/test_qwen4_exp_weight_loading.py::test_loads_non_unit_ple_embedding_weight_scale
AssertionError: Expected 1.0, got 0.00019931793212890625 in skipped model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale
1 failed
```

Restoring the prepared source produces:

```text
3 passed, 17 warnings in 12.36s
```

## GPU check

The actual loader helper copied the reported CPU FP32 checkpoint value to a
GPU-resident BF16 destination. The comparison reference was an independent CPU
BF16 conversion.

```text
device_name= AMD Instinct MI350X
gcn_arch= gfx950:sramecc+:xnack-
handled= True
gpu_value= 0.00019931793212890625
cpu_bf16_reference= 0.00019931793212890625
equal= True
marked_loaded= True
```

## Limitations

The reported Qwen 3.8 checkpoint weights were not present. Consequently this
does not claim a full model load, HTTP serving, semantic-accuracy, or
distributed-workload reproduction. The deterministic tiny Llama fixture is not
architecture-qualified for this Qwen-specific checkpoint loader failure, so it
was not substituted for the missing model.

No native C++ or FlyDSL source changed, and no native rebuild was required.
