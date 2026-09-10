# MI300X row-order capture/replay evidence

This slice tests permuted batch rows and inverse row order across repeated bounded capture/replay cycles on one AMD Instinct MI300X (`gfx942`).

## Result

No mismatch was demonstrated. `CompactRowIndex`, `CompactVerifyIds`, and `ScatterCompactToStrided` all passed four reused-graph cycles for the base, permuted, inverse, and reverse row orders.

## Reproduction

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
export TRITON_CACHE_DIR=/tmp/sglang-cache-j-74a5e0c24583/triton
export TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-74a5e0c24583/torchinductor
timeout 240 /opt/venv/bin/python reports/j-74a5e0c24583/probe_row_order_batch8.py
```

Expected output:

```text
cycle=1 pattern=base row_ok=True ids_ok=True scatter_ok=True
cycle=2 pattern=perm row_ok=True ids_ok=True scatter_ok=True
cycle=3 pattern=inverse row_ok=True ids_ok=True scatter_ok=True
cycle=4 pattern=reverse row_ok=True ids_ok=True scatter_ok=True
BATCH8_ROW_ORDER_PROBE_OK
```

## Notes

- The installed-source baseline is recorded in `/job/baseline-first.json`.
- The first `CompactVerifyIds` reuse probe produced a false mismatch because the reference indexed draft inputs by the original request ID rather than the new row position. The corrected reference passes all cycles.
- No production code was changed because no mismatch remained after correcting the reference.
