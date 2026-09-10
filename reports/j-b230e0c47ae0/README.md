# MoRI local stage-two reduction investigation

## Conclusion

The installed AITER FlyDSL masked stage-two reduction is correct on one MI300X for synthetic dispatch capacities 32, 64, 128, and 256. Every GPU result exactly matches an independent CPU/NumPy reference after bf16 rounding, all outputs are finite, invalid slots retain their sentinel, and both guard regions remain unchanged.

The historical masked-slot defect is already fixed by ROCm/aiter pull 3377, commit `7a8ff7dd4ae3063ff1a18622a46460125c84370e`. That commit is an ancestor of the installed AITER head `c16d44b93a528b2a4bfd6d8d3409116d465872a9`, so this change does not duplicate the kernel fix.

Baseline SGLang accepted capacities below the reported wave64 minimum. The minimal candidate adds a 64-token lower bound and rejects invalid values in the dispatcher constructor and before `mori.ops.EpDispatchCombineConfig` construction. Capacity 32 is therefore rejected even though the fixed local reduction kernel handles it correctly; this preserves the reported wave64 assumption rather than inferring a new kernel minimum.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, warp size 64, serial `692440003945`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Python: `/opt/venv/bin/python` 3.10.12; Torch `2.9.1+rocm7.2.0.git7e1940d4`.
- SGLang source: `/sgl-workspace/sglang/python/sglang`; delivery base `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- AITER source: `/sgl-workspace/aiter/aiter`; native module `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- MoRI source: `/sgl-workspace/mori/python/mori`.
- Full machine and path details are in `results/environment.json`.

## Native isolation

The probe calls AITER's private `_run_moe_reduction` helper directly from `/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py`. This isolates the stage-two reduce/indexing path without initializing MoRI communication or launching an EP8 job.

For each capacity, the probe uses:

- `token_num = capacity`, `topk = 8`, `model_dim = 128`.
- Eight global experts with experts 0 and 1 marked local.
- A flat bf16 dispatch buffer with 128-element sentinel guards on both sides.
- Deterministic valid slot values and a `-7777` sentinel in every masked slot (represented as `-7776` in bf16).
- An independent CPU/NumPy masked-sum reference.

## Results

| Capacity | Valid/invalid slots | Exact bf16 | Max float32 diff | Guards | Invalid-slot sentinel |
|---:|---:|---|---:|---|---|
| 32 | 81/175 | PASS | 0 | unchanged | unchanged |
| 64 | 175/337 | PASS | 1 | unchanged | unchanged |
| 128 | 346/678 | PASS | 2 | unchanged | unchanged |
| 256 | 707/1341 | PASS | 3 | unchanged | unchanged |

The nonzero float32 differences at 64, 128, and 256 are expected bf16 output rounding. The probe separately checks exact equality after casting the independent reference to bf16; all four capacities pass. Raw per-capacity output, hashes, and guard checksums are in `results/mori_reduce_results.json`.

## Capacity validation

Baseline `get_ep_dispatch_configs` accepted `-1`, `0`, `31`, `32`, `63`, `64`, `128`, and `256`; the raw baseline record is in `results/capacity_acceptance.json`. The candidate now:

- Accepts 64, 128, and 256.
- Rejects `-1`, `0`, 32, and 63 with a clear `ValueError`.
- Rejects capacity 32 before importing MoRI or constructing `mori.ops.EpDispatchCombineConfig`.

## Reproduction

Use a job-private cache outside the repository:

```bash
CACHE=/tmp/sglang-cache-JOB_ID
mkdir -p "$CACHE/flydsl" "$CACHE/aiter"
export AITER_FLYDSL_CACHE_DIR="$CACHE/flydsl"
export AITER_CACHE_DIR="$CACHE/aiter"
export MORI_REDUCE_RESULTS="$CACHE/mori_reduce_results.json"
/opt/venv/bin/python reports/j-b230e0c47ae0/mori_local_reduce_probe.py

export PYTHONPATH="$PWD/python"
/opt/venv/bin/python -m unittest -v \
  python.sglang.test.srt.layers.moe.token_dispatcher.test_moriep_dispatch_capacity
```

## Explicitly unsupported claims

- This one-GPU probe does not exercise EP8 communication, RDMA, MoRI dispatch/combine, or multi-rank routing.
- No model weights were downloaded and no GSM8K evaluation was run.
- These results do not establish, reproduce, or restore any distributed GSM8K accuracy number.
- The fixed AITER native path passing at capacity 32 does not override the reported wave64 dispatch-buffer assumption.
- The historical AITER defect was not reproduced because the installed source already contains the cited fix.
