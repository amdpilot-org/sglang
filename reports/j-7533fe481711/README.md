# Breakable DSA paging initialization report

## Result

Current `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c` reproduces the missing ROCm breakable-graph DSA paging dispatch in a reduced, no-weights runner/backend fixture. Eager and full-graph controls pass. Breakable capture fails before the paging object is available:

```text
AssertionError: Internal error: in-graph DSA prefill must go through the graph DSA split-op dispatch
```

Upstream PR `sgl-project/sglang#35279`, tested at commit `4d02d1f81107904eca0d096c772b512b2a78f4a3`, passes the same reduced fixture on one MI300X. Its three changes remove the CUDA-only split-op gates and enable the same dispatch on HIP. This report does not duplicate that working candidate.

The fixture checks that the paging object is non-None before calling `get_page_table_64`, captures/replays a small GPU page table, and compares readback with an independent Python indexing reference. Initial page IDs are `[7, 8, 9, 10, 11, 12]`; replay page IDs are `[31, 32, 33, 34, 35, 36]`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, gfx942, 304 CUs, serial `692440003936`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Current source: `/job/sglang`, commit `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Candidate worktree: `/tmp/sglang-cache-j-7533fe481711/candidate-35279`, commit `4d02d1f81107904eca0d096c772b512b2a78f4a3`
- Current Python source import: `/job/sglang/python/sglang/__init__.py`
- DSA indexer import: `/job/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- TileLang DSA kernel import: `/job/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py`
- Torch native package: `/opt/venv/lib/python3.10/site-packages/torch`
- `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`
- TileLang: `/opt/tilelang/tilelang`
- AITER: `/sgl-workspace/aiter/aiter`
- `sglang._C` is unavailable in this environment; the fixture uses the installed `sgl_kernel`, TileLang, and AITER native stack.

## Reproduction

The fixture is retained as `reduced_dsa_paging_fixture.py`. Run it with the current source:

```bash
SGLANG_SOURCE=/job/sglang \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python reports/j-7533fe481711/reduced_dsa_paging_fixture.py
```

Run the tested upstream candidate from a worktree at `4d02d1f81107904eca0d096c772b512b2a78f4a3`:

```bash
SGLANG_SOURCE=/tmp/sglang-cache-j-7533fe481711/candidate-35279 \
PYTHONPATH=/tmp/sglang-cache-j-7533fe481711/candidate-35279/python \
/opt/venv/bin/python /tmp/sglang-cache-j-7533fe481711/reduced_dsa_paging_fixture.py
```

The current-source command exits nonzero because breakable capture fails. The candidate command exits zero with all three modes passing.

## Raw results

Raw JSON is retained in `current-final-result.json` and `candidate-35279-result.json`.

| Source | Eager | Full graph | Breakable |
|---|---|---|---|
| Current `0084030179` | pass | pass | fail before paging dispatch |
| Candidate `4d02d1f811` | pass | pass | pass |

All passing readbacks equal the independent Python-indexed reference. The only numerical values checked are the six initial and six replay page IDs above; no model weights were downloaded and no numerical gate was changed.

## Untested and uncertain

- No GLM checkpoint was used.
- No PP/TP server or multi-rank serving was run; multi-rank serving remains untested.
- This reduced fixture validates paging dispatch/readback, not full DSA model accuracy or end-to-end serving throughput.
- Upstream PR `sgl-project/sglang#35279` had not received normal CI because its required `run-ci` label was absent at the time of inspection.
