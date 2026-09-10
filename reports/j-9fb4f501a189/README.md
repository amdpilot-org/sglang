# gfx942 scalar metadata sharing and true-shape cache separation

## What I did

This is a report-only follow-up to `amdpilot-org/sglang` issue 207 and upstream
`sgl-project/sglang` issue 31568. I did not repeat the already fulfilled basic
`N=17` versus `N=33` sharing scope from mirror pull request 246.

- Recorded an early installed-source baseline before cloning or editing.
- Cloned this mirror at `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Confirmed the same mismatch on mirror `main`.
- Tested the exact upstream candidate commit
  `8645d578414c6b4e06d7db30089c8b631c20f4ec` from `sgl-project/sglang` pull
  request 31689.
- Used one assigned AMD Instinct MI300X (`gfx942`) and the qualified
  Torch/ROCm stack.
- Used a finite adversarial matrix with independently derived PyTorch
  references and unchanged numerical gates.
- Verified that the two scalar metadata values `N=17` and `N=33` share one
  compiled cache entry on the candidate while a genuine `H/D` shape change still
  creates a separate cache entry and preserves exact output.
- Did not duplicate the already-working upstream kernel fix, so this PR contains
  only the investigation report and raw results.

## Reproduce

The installed-source baseline was recorded with:

```bash
TRITON_CACHE_DIR=/tmp/sglang-cache-j-9fb4f501a189/baseline \
/opt/venv/bin/python /tmp/sglang-baseline-9fb4f501a189.py
```

The mirror `main` control was recorded with:

```bash
TRITON_CACHE_DIR=/tmp/sglang-cache-j-9fb4f501a189/mirror-main \
PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python /tmp/sglang-mirror-baseline-9fb4f501a189.py
```

The upstream candidate was recorded with:

```bash
TRITON_CACHE_DIR=/tmp/sglang-cache-j-9fb4f501a189/candidate \
PYTHONPATH=/job/sglang-pr-head/python \
/opt/venv/bin/python /tmp/sglang-candidate-9fb4f501a189.py
```

All three used the same finite adversarial matrix:

- `N=17`, `H=2`, `D=8`, alternating mask
- `N=33`, `H=2`, `D=8`, alternating mask
- `N=17`, `H=3`, `D=5`, all-true mask

Timing used `time.perf_counter()` around one cold launch followed by
`torch.cuda.synchronize()`, then ten bounded warm launches with the median
reported. The independent reference assigned selected rows with ordinary PyTorch
indexing and asserted that masked-out rows remained untouched.

The complete early installed-source record is also saved outside the repository
at `/job/baseline-first.json`; the PR copy is in `results.json`.

## Results

| Source | `N` | `H` | `D` | Cache entries | Cold ms | Warm median ms | Max K/V error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Installed source | 17 | 2 | 8 | 1 | 816.955 | 0.040828 | 0 / 0 |
| Installed source | 33 | 2 | 8 | 2 | 39.663 | 0.038318 | 0 / 0 |
| Installed source | 17 | 3 | 5 | 3 | 39.570 | 0.046666 | 0 / 0 |
| Mirror `main` | 17 | 2 | 8 | 1 | 801.037 | 0.039494 | 0 / 0 |
| Mirror `main` | 33 | 2 | 8 | 2 | 39.387 | 0.038471 | 0 / 0 |
| Mirror `main` | 17 | 3 | 5 | 3 | 36.533 | 0.037887 | 0 / 0 |
| Upstream candidate | 17 | 2 | 8 | 1 | 831.079 | 0.038940 | 0 / 0 |
| Upstream candidate | 33 | 2 | 8 | 1 | 0.183 | 0.038153 | 0 / 0 |
| Upstream candidate | 17 | 3 | 5 | 2 | 40.578 | 0.046329 | 0 / 0 |

The candidate’s `N=17` and `N=33` launches share the same cache key. The
genuine `H/D` shape change creates a second, distinct cache key. All selected
and untouched output assertions remain exact.

The candidate’s committed regression also passed on `gfx942`:

```text
1 passed, 3 warnings in 15.93s
```

## Unsupported boundary

The XPU FLA `NT_BUCKET` axis was not executed on `gfx942` because
`torch.xpu.is_available()` returned `False`. Static inspection confirms that
`NT_BUCKET` appears only in the autotune key, kernel signature, and call-site
bucket calculation, not in the kernel body. The supported neighboring control
is the masked KV write kernel above.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: AMD Instinct MI300X, `gfx942`, GUID `47961`
- Python: `/opt/venv/bin/python`
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP library: `/opt/venv/lib/python3.10/site-packages/torch/lib/libtorch_hip.so`
- Triton: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`
- Installed SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Mirror SGLang: `/job/sglang/python/sglang/__init__.py`
- Candidate SGLang: `/job/sglang-pr-head/python/sglang/__init__.py`

No upstream issue, pull request, or comment was posted or modified.
