# gfx942 top-k permutation-invariance investigation

## Conclusion

The property is real on the current mirror `main` revision, and upstream PR
37941 already fixes it on one assigned MI300X. I made no code change because
the tested candidate commit is already working and the task explicitly says
not to duplicate a fulfilled identical scope.

- Current mirror `main`: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- Tested upstream candidate: `7a83d7fdd548caf1ed4da9525e1a17724661172c`
- Upstream PR: `sgl-project/sglang#37941`
- Reference issue: `sgl-project/sglang#35257`
- Prior mirror issue: `amdpilot-org/sglang#221`

## Property and finite adversarial matrix

The metamorphic property is:

> Permuting the positions of equal-score candidates must not cause the kernel
> to drop any strictly larger score from the valid top-k membership.

The finite matrix uses:

- `k = 2048`
- 2,500 or 3,000 equal scores at `1.0`
- 100 strictly larger scores at `1.0001`
- The remaining scores are `0.0`
- The equal-score positions are permuted across a bounded number of seeds
- The independent reference is `torch.topk(..., sorted=False).indices`

The strictly larger positions are fixed while the equal-score positions move.
The oracle accepts any valid subset of the equal-score group, but requires all
100 strictly larger positions to remain in the output.

## gfx942 results

| Revision | Path | Permutations | Dropped larger scores | Reference mismatch |
|---|---|---:|---:|---:|
| `0084030` | Register, batch 6, seq 8192 | 1 | 22 | 222 |
| `0084030` | Streaming, batch 4, seq 32768 | 1 | 36 | 63 |
| `0084030` | ROCm boundary, batch 1, seq 65537 | 1 | 25 | 55 |
| `7a83d7f` | Register, batch 6, seq 8192 | 32 | 0 | 0 |
| `7a83d7f` | Streaming, batch 4, seq 32768 | 16 | 0 | 0 |
| `7a83d7f` | ROCm boundary, batch 1, seq 65537 | 8 | 0 | 0 |

The candidate also passes:

- 8 focused overflow tests in `test_topk_v2.py`
- The complete focused top-k suite: 286 passed in 32.54 seconds

## Commands

The installed-source baseline and all later runs used:

```bash
export PYTHONPATH=/job/sglang/python
export SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-630d4dadb380
/opt/venv/bin/python -m pytest test/registered/kernels/ops/attention/test_topk_v2.py -q
```

The custom metamorphic runs used the same interpreter, GPU, and JIT cache, with
`plan_topk_v2` and `topk_transform_paged_v2` from the checked-out source. Raw
results are in `results.json`.

## Environment and paths

- GPU: one AMD Instinct MI300X, gfx942
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Interpreter: `/opt/venv/bin/python`
- Installed source: `/sgl-workspace/sglang/python/sglang`
- Installed native module: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`
- Mirror source: `/job/sglang/python/sglang`
- Job-private JIT cache: `/tmp/sglang-cache-j-630d4dadb380`

## Unsupported and unverified boundaries

- The installed-source v2 JIT path could not compile because
  `cooperative_groups.h` was missing.
- The installed AOT v1 path faulted at `seq=65537`.
- The mirror checkout compiles on ROCm because the include is guarded.
- PR 37941 still leaves the CUDA-only cluster path truncating; that path is not
  compiled on gfx942 and was not exercised here.
- The requested image ID could not be independently re-read because neither
  `docker` nor `podman` is available in the job container.

## Numerical gates

No numerical gate was changed. The pre-existing suite keeps
`MAX_PERMIT_ERROR = 5`; the candidate’s new overflow tests use an exact
`max_permit_error = 0` gate. The custom oracle also requires exact membership
for strictly larger scores while allowing arbitrary equal-score substitutions.
