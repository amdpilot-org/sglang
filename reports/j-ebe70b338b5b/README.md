# Repeated Mamba verify/reset destination investigation

## Scope

This report covers the additional property that state-tracking indices across
repeated `TARGET_VERIFY` / reset transitions must not reuse stale destinations.
It is intentionally separate from the already-covered absent-index and
single-transition metadata scope in mirror issue 216 and PR 243.

No production mismatch was demonstrated, so no production code was changed.

## Result

On one assigned AMD Instinct MI300X (`gfx942`), a finite 16-case adversarial
matrix performed three verifies through two real `ScheduleBatch.filter_batch`
resets per case:

- 48 total verify transitions.
- 32 total reset transitions.
- Every destination was compared with an independently derived direct Python
  `mapping[request_index, plan]` reference.
- Every case passed, including changed request rows, changed ping-pong plans,
  and disjoint destination ranges after each reset.
- The existing focused GPU suite also passed: 31 tests, including the unchanged
  fused/reference numerical gates.

The complete case matrix and timing are in `mirror-gpu-results.json`.

## Reproduction

From the delivery checkout:

```bash
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-ebe70b338b5b
/opt/venv/bin/python reports/j-ebe70b338b5b/repeated_verify_reset_probe.py
/opt/venv/bin/python -m pytest -q -s -p no:cacheprovider \
  test/registered/kernels/ops/attention/test_fused_verify_triton_gdn.py \
  test/registered/unit/layers/test_mamba_state_scatter_triton.py \
  test/registered/unit/spec/test_ngram_mamba_verify_update.py
```

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440004395`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local
  image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Delivery checkout: `/job/sglang`, branch
  `amdpilot/j-ebe70b338b5b`, base commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Job-private Triton cache: `/tmp/sglang-cache-j-ebe70b338b5b`.

The installed-source baseline is recorded separately at
`/job/baseline-first.json`; it is not evidence for checkout changes.

## Limitations

- This is a focused metadata and kernel-contract result, not a full-model E2E run.
- No model weights were downloaded.
- Only one MI300X (`gfx942`) was used; NVIDIA and multi-GPU behavior are not established.
- The probe intentionally uses tiny synthetic state mappings and does not exercise
  full scheduler admission or model execution.

## Harness notes

Two intermediate probe failures were harness-only and were corrected before the
final run:

- The first run omitted the explicit `extra_buffer_lazy` server-args override, so
  the runtime execution namespace was not published.
- A later run derived an invalid `-1` ping-pong plan for one filtered row and
  caused an out-of-bounds HIP gather. The corrected probe always derives valid
  `0`/`1` plans.

Neither intermediate failure was treated as a production mismatch.
