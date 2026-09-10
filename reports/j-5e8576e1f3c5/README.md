# One-GPU gfx942 reduced DSpark block study

This report is a decode-sized latency study built from existing SGLang primitives:
target hidden-state selection, LM-head projection, a recurrent Markov draft block,
greedy target acceptance, and output-token commit. It uses locally generated synthetic
weights and states only. It does not use GLM weights, run a distributed configuration,
or claim workload-independent speedup.

Run from the repository root:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-5e8576e1f3c5/reduced_pipeline.py \
  --output reports/j-5e8576e1f3c5/results.json
```

The harness runs six small batch/gamma cases, four real block forwards per recurrent
sequence, five warmup sequences, and thirty measured sequences per case. It validates
each handoff against an independent reference after every recurrent step and records
median, p90, p99, minimum, and maximum latency. Exact dimensions, dtypes, gates,
memory, source/native paths, and raw timings are in `results.json`.

The model-specific constructor boundary is explicit: no GLM or DeepSeek model-specific
DSpark constructor is instantiated. The study uses the generic `VanillaMarkov` head and
a synthetic LM-head adapter. This does not validate a full model-specific DSpark
implementation.

## Installed-source first GPU baseline

Before cloning or editing, the qualified image ran the installed AOT speculative tests:

```bash
/opt/venv/bin/python -m pytest -q \
  /sgl-workspace/sglang/python/sglang/kernels/aot/tests/speculative/test_eagle_utils.py \
  /sgl-workspace/sglang/python/sglang/kernels/aot/tests/speculative/test_speculative_sampling.py
```

The first GPU execution completed in 7 seconds. `verify_tree_greedy` passed its
independent expected outputs; `tree_speculative_sampling_target_only` was unavailable
in the installed native build with:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute
'tree_speculative_sampling_target_only'
```

A bounded supported control used 20 warmup calls and 100 measured CUDA-event calls of
`verify_tree_greedy`: median 0.020066 ms, p90 0.022291 ms. This installed-source result
is labeled separately in `/job/baseline-first.json` and is not proof for checkout changes.

## Checkout validation and results

The persistent mirror checkout is `amdpilot/j-5e8576e1f3c5` at base commit
`0084030179bfba86bfeb6d43f7997d4076329d2c`. The checkout validation command was:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/spec/dspark/test_dspark_kernel_parity.py \
  test/registered/spec/utils/test_build_eagle_tree.py
```

It passed 1 DSpark parity test with 22 subtests and 2 tree-layout tests. The reduced
study completed 840 real block forwards (including warmup) in six cases. Generated
weights were 1,572,864 bytes; final live allocation was 81,265,664 bytes and peak
allocation was 84,369,920 bytes.

| Case | Batch | Gamma | Median 4-step ms | p90 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 2 | 2.5471 | 2.6929 | 2.9221 |
| 1 | 1 | 4 | 3.4956 | 3.8130 | 3.8546 |
| 2 | 2 | 4 | 3.5330 | 3.8268 | 7.0601 |
| 3 | 4 | 4 | 3.5578 | 3.7911 | 3.9016 |
| 4 | 8 | 4 | 3.5711 | 3.8645 | 4.0074 |
| 5 | 4 | 8 | 5.7127 | 6.0679 | 6.4722 |

No upstream issue, PR, or comment was posted or changed. No upstream candidate PR was
checked out because this delivery adds a bounded report rather than duplicating a fix;
the tested source commit is preserved in `results.json`.
