# gfx942 page-split factorization control

## Scope

- Campaign: `repo-e2e-20260909`; job: `j-54154d37acc6`.
- Distinct follow-up to `amdpilot-org/sglang` issue 220: equivalent byte offsets expressed through different safe shape/stride factorizations must agree.
- Read-only upstream context: `sgl-project/sglang` issue 34025 and pull request 34027.
- Existing mirror pull request 248 already fulfills issue 220’s int32 lane-offset scope; this control does not repeat or duplicate that fix.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Interpreter: `/opt/venv/bin/python` (`3.10.12`).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, Python path `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`.
- Triton Python path: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- Installed `sglang` import path: `/sgl-workspace/sglang/python/sglang/__init__.py`.
- Installed `sgl_kernel` path: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`.
- Installed `aiter` path: `/sgl-workspace/aiter/aiter/__init__.py`.
- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440004359`, node ID 3, GUID 39656.
- Mirror base: `0084030179bfba86bfeb6d43f7997d4076329d2c` (`main`).
- Open fix candidate commit tested: `fc0ae60bf07af8f141be9debf8fc07153bccec00` (`amdpilot/j-c25addb40591`, mirror PR 248).

## Commands

All runs used `PYTHONPATH=<checkout>/python`, `/opt/venv/bin/python`, and a job-private Triton cache under `/tmp/sglang-cache-j-54154d37acc6/`. Timing used one-shot `date +%s.%N` before and after each invocation; no burn, unbounded loops, sleeps, or repeated work.

### Installed-source baseline

```bash
PYTHONPATH=/sgl-workspace/sglang/python timeout 180 /opt/venv/bin/python -m pytest -q -s \
  /sgl-workspace/sglang/test/registered/kernels/ops/attention/test_flash_mla_backends.py::TestTouchedPageSplit::test_only_marked_pages_are_split
```

Result: `1 skipped`, 15.967 s wall. The class is gated by `torch.cuda.get_device_capability() == (12, 0)`; `gfx942` does not satisfy it.

```bash
PYTHONPATH=/sgl-workspace/sglang/python timeout 180 /opt/venv/bin/python -m pytest -q -s \
  /sgl-workspace/sglang/test/registered/kernels/ops/moe/test_shuffle_rows_with_scales.py::test_matches_torch_gather[512-1-8]
```

Result: `1 passed`, 7.100 s wall. The JIT row gather matched an independent torch advanced-indexing reference bit-exactly.

Full baseline metadata is saved outside the repository at `/job/baseline-first.json`; it is installed-source context and is not proof for later checkout changes.

### Factorization control

The control allocated two 149,760-byte source pages (299,520 bytes total) and derived a deterministic adversarial byte matrix with `torch.arange(...) % 251`. It compared `_split_kv_pages_to_64` outputs for these equivalent views of the same storage:

- 2D `(2, 149760)`, stride `(149760, 1)`.
- 3D `(2, 256, 584)`, stride `(149760, 584, 1)`.
- 4D standard `(2, 256, 1, 584)`, stride `(149760, 584, 584, 1)`.
- 4D alternate `(2, 1, 256, 584)`, stride `(149760, 149760, 584, 1)`.

The independent reference copied each 36,864-byte data sub-page and 512-byte scale sub-page directly from the 2D source matrix into the expected 37,440-byte destination rows. The numerical gate was unchanged: exact `torch.equal` on the copied data+scale region, with maximum absolute byte difference 0.

Representative command:

```bash
PYTHONPATH=/job/sglang/python TRITON_CACHE_DIR=/tmp/sglang-cache-j-54154d37acc6/triton \
  /opt/venv/bin/python - <<'PY'
# Build the four equivalent views above, call _split_kv_pages_to_64(view, 256),
# flatten each returned view to (8, 37440), and compare the first 37,376 bytes
# against the independently derived source slices with torch.equal.
PY
```

## Results

- Mirror `main` first factorization pair: 4D and 2D outputs equal; both equal the independent reference; maximum difference 0; 5.990 s wall.
- Mirror `main` widened set: 2D, 3D, standard 4D, and alternate 4D all equal the independent reference and each other; maximum difference 0; 7.789 s wall.
- Open fix candidate `fc0ae60bf07af8f141be9debf8fc07153bccec00`: all four factorizations equal the independent reference and each other; maximum difference 0; 6.167 s wall.
- No mismatch was demonstrated, so no operation code was changed.

## Limitations

- The assigned GPU is MI300X/`gfx942`, not SM120; the production SM120 FlashMLA dispatch was not run end-to-end.
- The relevant installed SM120 page-split test is architecture-gated and skipped on `gfx942`; the supported neighboring control and direct Triton kernel execution provide the meaningful `gfx942` evidence.
- No full model weights, multi-million-token KV pool, invalid wrapped-address dereference, environment replacement, or node-wide state change was used.
- No upstream issue, pull request, or comment was posted or modified.

## Conclusion

Equivalent byte offsets expressed through the tested safe 2D, 3D, standard 4D, and alternate 4D shape/stride factorizations agree exactly on one `gfx942` GPU, both on mirror `main` and on the open issue-220 fix candidate. This is a negative result for the additional property: no code change is warranted.
