# ROCm DSA/IndexerKPool dual-stream gate investigation

## Conclusion

The HIP zero threshold is best read as an **intentional unsupported path**, not a
lost capability that should be blindly enabled.

Evidence:

- The original NSA implementation introduced `1024 if CUDA else 0` together with
  the `0 < tokens <= threshold` predicate in commit
  `efbc687c28176faa36a5c3872a470bd7ba096951`.
- Closed upstream PR 28846 explicitly documented the HIP value as disabled:
  enabling indexer dual-stream overlap was neutral at MI355X TP4 and risked the
  TP8 hardware-queue-oversubscription regression. That PR was closed without
  merging, but its rationale is direct evidence of intent.
- Current upstream `sgl-project/sglang` main (`3700c4ee26a1df3fd27e10a4a83d40d991d87d6c`)
  still contains the same zero threshold and contradictory predicate.
- On this MI300X control, the underlying KPool primitive can use an alternative
  stream and can be captured/replayed after a diagnostic-only attribute is added.
  However, normal HIP construction omits `half_device_sm_count`, which the forced
  dual-stream branch reads. That makes the current HIP path incomplete/unsupported
  even before the threshold is considered.

No runtime gate or threshold was changed. This MI300X result must not be used to
infer MI355X TP8 behavior or to justify enabling concurrency.

## Runtime results

The synthetic input was eight tokens. The real model wiring was exercised with
`SGLANG_ROCM_USE_MULTI_STREAM` off and on.

| Runtime option | Model alt stream | Indexer alt stream | Runtime gate | Alt-stream contexts |
|---|---:|---:|---:|---:|
| `0` | absent | absent | `false` | 0 |
| `1` | allocated | allocated | `false` | 0 |

With the option on, the gate remains false because
`DUAL_STREAM_TOKEN_THRESHOLD == 0` makes `tokens > 0 && tokens <= 0`
unsatisfiable.

### Diagnostic forced primitive

For capability diagnosis only, the option-on run supplied
`indexer.half_device_sm_count = 64` and called `_get_q_k_bf16(..., True)`
directly. This did not change the runtime gate.

- Alternative-stream contexts: 1.
- Query equality with the single-stream reference: exact, max absolute difference 0.
- Key equality with the single-stream reference: exact, max absolute difference 0.

### Graph capture and replay

The forced dual-stream primitive was captured with `torch.cuda.CUDAGraph` and
replayed after replacing the static input values.

- Capture used one alternative-stream context.
- Captured query/key outputs exactly matched the single-stream reference.
- Replay query/key outputs exactly matched a fresh single-stream reference on the
  new inputs; max absolute difference was 0 for both.

This proves the small projection primitive is graph-capable on gfx942 when given
the missing diagnostic attribute. It does not prove that the full DSA/KPool runtime
path is supported or beneficial on HIP.

## Environment

- Qualified local image ID:
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
  (operator-provided local image ID; not a pullable registry digest).
- Hostname: `banff-cyxtera-cx57-4` (not image identity).
- OS: Ubuntu 22.04.5 LTS.
- GPU: one AMD Instinct MI300X, gfx942, serial `692440004306`, unique ID
  `0xb5c590cf4c10631d`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`.
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Current upstream main checked read-only:
  `3700c4ee26a1df3fd27e10a4a83d40d991d87d6c`.

### Paths

- Python: `/opt/venv/bin/python`.
- Torch Python module:
  `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Torch HIP library:
  `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`.
- SGLang Python source:
  `/job/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py` and
  `/job/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py`.
- AITER Python module: `/sgl-workspace/aiter/aiter/__init__.py`.
- AITER native module:
  `/sgl-workspace/aiter/aiter/jit/build/module_aiter_core/build/module_aiter_core.so`.
- `sgl_kernel` Python module:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- `sgl_kernel` native module:
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.

## Reproduction

The diagnostic script is
`reports/j-6f211c082ab8/probe_gate_graph.py`. It uses a tiny one-layer
DeepSeek V2 model with an `IndexerKPool`, bounded eight-token inputs, and local
single-rank parallel stubs. It records the real model option wiring, evaluates the
runtime gate, compares forced dual-stream outputs with a single-stream reference,
and captures/replays the forced primitive.

Run:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python \
SGLANG_ROCM_USE_MULTI_STREAM=0 \
/opt/venv/bin/python reports/j-6f211c082ab8/probe_gate_graph.py

PYTHONPATH=/job/sglang/python \
SGLANG_ROCM_USE_MULTI_STREAM=1 \
/opt/venv/bin/python reports/j-6f211c082ab8/probe_gate_graph.py
```

Raw output is in `reports/j-6f211c082ab8/logs/gate-graph.log`.

The existing adjacent test was also attempted:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest \
  test/registered/kernels/ops/attention/test_dsa_indexer.py \
  -k alt_stream -q
```

It failed before exercising the stream because the test patches
`sglang.srt.layers.attention.dsa.dsa_indexer.deep_gemm`, which is absent on HIP.
The raw failure is in
`reports/j-6f211c082ab8/logs/existing-alt-stream-test.log`. This is an existing
ROCm test incompatibility, not evidence about the dual-stream gate.

## Upstream context

- `sgl-project/sglang` issue 38745 reports the unsatisfiable HIP predicate and
  had no comments when read.
- PR 14337 removed token thresholds from several other models and left the DSA/NSA
  TODO explicitly unresolved.
- PR 9405 changed DeepSeek MoE to use dual stream whenever CUDA graph capture is
  enabled, but did not address this DSA/IndexerKPool gate.
- Closed PR 28846 added the clearest intent comment: HIP keeps the threshold at
  zero because the overlap was neutral at TP4 and risks TP8 hardware-queue
  oversubscription. Its exact diff hunk is preserved in
  `reports/j-6f211c082ab8/logs/pr28846.diff`.

## Limitations

- This is a bounded synthetic eight-token control, not a full-model benchmark.
- No model weights were downloaded.
- No performance claim is made from this MI300X run.
- No MI355X or TP8 behavior was tested or inferred.
- The forced primitive required a diagnostic-only `half_device_sm_count`; the
  production HIP constructor does not provide it.
