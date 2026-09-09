# ROCm DSpark top-k renormalization dispatch investigation

## Result

On the assigned MI300X (gfx942), current mirror `main` commit `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` does **not** resolve `top_k_renorm_prob` to `None`. It resolves to `sglang.kernels.ops.sampling.renorm_triton.top_k_renorm_probs_triton`. The same is true for `top_p_renorm_prob`.

The issue-era control commit `60d6914f17ec691edb3258601a3c65b1e4f8de61` resolves both names to `None` on HIP because its dispatch only attempts `sgl_kernel` when `is_cuda() or is_musa()` is true. On this ROCm stack, `is_cuda()` is false (`torch.version.cuda is None`) even though `torch.cuda.is_available()` is true, while `is_hip()` is true.

This is an investigation report only. It does not claim that any closed or open candidate should be merged.

## Environment

- GPU: one AMD Instinct MI300X, gfx942, serial `692440004359`, node ID 3.
- Image: operator-specified local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1` (`amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`). Container hostname was not used as image identity.
- OS: Ubuntu 22.04.5 LTS.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`; `hipcc` reports ROCm 7.2.0 / AMD clang 22.0.0git.
- Triton: `/opt/venv/lib/python3.10/site-packages/triton/__init__.py`.
- Installed `sgl_kernel`: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`; sampling wrapper `/opt/venv/lib/python3.10/site-packages/sgl_kernel/sampling.py`; native library `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.
- Working clone: `/job/sglang`, cloned from `https://github.com/amdpilot-org/sglang.git`, delivery branch `amdpilot/j-c8a20ab909bd`.
- Current source: `/job/sglang/python/sglang/srt/speculative/dflash_utils.py` and `/job/sglang/python/sglang/kernels/ops/sampling/renorm_triton.py`.
- Candidate worktrees: `/job/candidate-issue-base`, `/job/candidate-32621`, `/job/candidate-33301`, and `/job/candidate-32630`.
- Caches were kept job-private under `/job/.triton-cache`, `/job/.cache/hf`, and `/job/.cache/torch`. No model weights were downloaded.

## Dispatch conditions

Current `main` uses this import-time dispatch:

1. If `is_cuda() or is_musa()`, try importing `top_k_renorm_prob`, `top_p_renorm_prob`, and `tree_speculative_sampling_target_only` from `sgl_kernel`. A successful import makes the two renorm names callable and sets `_DFLASH_SAMPLING_VERIFY_AVAILABLE = True`. An import exception makes all three names `None`.
2. Otherwise, if `is_hip()`, import both names from `sglang.kernels.ops.sampling.renorm_triton`. On current `main` this branch does not set `_DFLASH_SAMPLING_VERIFY_AVAILABLE`, so it remains `False`.
3. Otherwise, set both renorm names to `None`.

On gfx942, current `main` reports `is_cuda=False`, `is_hip=True`, `is_musa=False`, so it takes branch 2. Backend flags such as `SGLANG_USE_AITER`, `SGLANG_AITER_K3_OPT`, `AITER_FLYDSL_FORCE`, and `AITER_SITUV2_A8W4` are not consulted by this import-time dispatch.

Resolution is independent of tensor dtype and device. Once called, the current Triton backend requires a 2D CUDA/HIP tensor. It converts the input to contiguous `float32`. Direct checks produced:

- CPU 2D input: `ValueError: renorm kernels require a CUDA/HIP tensor`.
- HIP 1D input: `ValueError: probs must be 2D, got shape=(8,)`.

Current `main` also has `_dflash_top_k_renorm_prob` and `_dflash_top_p_renorm_prob` helpers. They call the imported backend when it is not `None`; otherwise they use NPU or Torch fallbacks. The DSpark verify builder can therefore exercise the real Triton dispatch even though `_DFLASH_SAMPLING_VERIFY_AVAILABLE` is false.

## DSpark/EAGLE request path

`SamplingBatchInfo.from_schedule_batch` sets:

- `need_top_p_sampling = any(request top_p != 1.0)`.
- `need_top_k_sampling = any(request top_k != TOP_K_ALL)`, where `TOP_K_ALL = 1 << 30`.

In `dspark_accept._accept_sampling_core`, neither flag causes `SoftmaxTemp.execute`. Either flag causes `build_dflash_verify_target_probs`. This matches the upstream issue observation that DSPARK can run until a request supplies `top_p < 1` or `top_k > 1`.

With the builder's default `use_sparse_topk=True`, a top-k request with `0 < max_top_k < vocab_size` uses a Torch `topk` sparse path. The dense renorm helpers are reached when that sparse path does not apply, or when `use_sparse_topk=False` is requested. The validator forces `use_sparse_topk=False` so the actual renorm dispatch is exercised.

The DFlash worker's separate non-greedy path is gated by `is_dflash_sampling_verify_available()`. On current `main` this is false on HIP, so that worker logs a warning and falls back to greedy argmax verification. DSpark's `AcceptSampling.execute` path calls the builder directly when top-k/top-p is requested and is not gated by that availability flag.

## Tested revisions

| Revision | Commit | `top_k_renorm_prob` on gfx942 | Outcome |
|---|---|---|---|
| Current mirror `main` | `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4` | Callable Triton backend | All synthetic cases pass the fixed gate. |
| Issue-era control | `60d6914f17ec691edb3258601a3c65b1e4f8de61` | `None` | Direct dispatch and builder fail with `TypeError: 'NoneType' object is not callable`. |
| Upstream PR 32621 head | `1305c2bbeb2d7c8728386a55d46f80be5df04423` | `None`/undefined | Top-p Triton passes exactly; top-k dispatch and builder fail with `NameError`. |
| Upstream PR 33301 head | `9c3e15e07968d61bf527a44543b398e88770d6f8` | Python wrapper is callable, native op is absent | Calls fail with `AttributeError` for missing `torch.ops.sgl_kernel` ops. |
| Upstream PR 32630 head | `7ce34e3c10420f27342a4491e70eb319a3a6403f` | Callable Torch fallback | Float32 builder passes exactly; direct half/bfloat16 cases fail the fixed gate because tie/order and output-dtype semantics differ. |

PR 32621 was merged upstream into the Kimi branch and then reached `main` through PR 32541. PR 33301 was open and unmerged at investigation time. PR 32630 was closed without merge. Upstream issue 32569 was open at investigation time. No upstream issue, PR, or comment was modified.

## Numerical gate and reference

The gate was fixed before comparisons and was not changed between revisions:

- Seed: `20260909`.
- Shapes/dtypes: `float32 [4,1024]`, `bfloat16 [3,4097]`, and `float16 [2,2048]`.
- Top-k values: `[1, 5, 17, min(64, vocab_size)]` as `int32`.
- Top-p values: `[0.25, 0.5, 0.8, 1.0]` as `float32`.
- Pass requires both maximum absolute element error and maximum absolute row-sum error to be `<= 2e-6`.

The independent Torch reference in `validate_renorm.py`:

- Top-k: sort descending, take the `(k - 1)` pivot, retain all values at or above the pivot, and renormalize.
- Top-p: sort ascending, compute the CDF, use `searchsorted(..., right=False)` for the FlashInfer-style pivot, retain all values at or above the pivot, and renormalize.

For backends returning a lower-precision dtype, the reference is cast to that output dtype before measuring. This does not hide tie-order differences because the reference remains independently computed.

## Raw observations

### Current `main`

All direct top-k, direct top-p, selected-helper top-k, selected-helper top-p, and `build_dflash_verify_target_probs` comparisons passed.

- Direct and selected renorm errors: `0.0` for all three shapes/dtypes.
- Builder maximum absolute error: `7.450580596923828e-09` for `float32 [4,1024]`, `1.4901161193847656e-08` for `bfloat16 [3,4097]`, and `0.0` for `float16 [2,2048]`.
- Builder maximum row-sum error: at most `5.960464477539063e-08`.

### Issue-era control

Both `top_k_renorm_prob` and `top_p_renorm_prob` were `None`. Every direct call was non-callable and every builder attempt failed with:

```text
TypeError: 'NoneType' object is not callable
```

### PR 32621

`top_p_renorm_prob` was the new Triton callable and matched the reference exactly for all cases. `top_k_renorm_prob` was absent, so the builder failed with:

```text
NameError: name 'top_k_renorm_prob' is not defined
```

This candidate does not provide a callable top-k backend on gfx942.

### PR 33301

Both names resolved to the installed `sgl_kernel` Python wrappers, so they were not `None`. Invocation failed because the qualified native Torch operations were absent:

```text
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'top_k_renorm_probs'
AttributeError: '_OpNamespace' 'sgl_kernel' object has no attribute 'top_p_renorm_probs'
```

Thus this candidate changes the failure from `NoneType` not callable to a callable-but-unsupported backend on this image.

### PR 32630

Both names resolved to portable Torch fallbacks. The builder passed exactly for all three cases because its input probabilities were `float32`. Direct `float32` renorm also passed exactly.

The fixed gate failed for direct `bfloat16` and `float16` inputs. Representative raw values:

- `bfloat16 [3,4097]` top-k: maximum absolute error `0.00048828125`, row-sum error `0.001220703125`.
- `bfloat16 [3,4097]` top-p: maximum absolute error `0.0031585693359375`, row-sum error `0.002410888671875`.
- `float16 [2,2048]` top-k: maximum absolute error `0.000244140625`, row-sum error `0.000244140625`.
- `float16 [2,2048]` top-p: maximum absolute error `3.0517578125e-05`, row-sum error `0.00035858154296875`.

The differences come from the candidate's rank-based top-k tie selection, descending-prefix top-p semantics, and lower-precision output, which differ from the FlashInfer-style tie-inclusive pivot reference. These are negative comparison results, not a claim that the candidate is unsuitable for every workload.

## Reproduction

The working clone was created with bounded retries:

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/amdpilot-org/sglang.git /job/sglang
cd /job/sglang
git checkout -B amdpilot/j-c8a20ab909bd origin/main
```

Issue and candidate metadata were read with:

```bash
gh issue view 32569 --repo sgl-project/sglang \
  --json number,title,state,author,body,comments,url
gh pr diff 32621 --repo sgl-project/sglang
gh pr diff 33301 --repo sgl-project/sglang
gh pr diff 32630 --repo sgl-project/sglang
```

Each validation run used the existing `/opt/venv/bin/python` and only changed `PYTHONPATH` to select the source worktree:

```bash
export TRITON_CACHE_DIR=/job/.triton-cache
export HF_HOME=/job/.cache/hf
export TORCH_HOME=/job/.cache/torch
export PYTHONPATH=/job/sglang/python

/opt/venv/bin/python \
  reports/j-c8a20ab909bd/validate_renorm.py \
  --label current-main \
  --commit ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4
```

The same command was run with `PYTHONPATH` pointed at each candidate worktree and the corresponding `--label` and `--commit`.

Raw JSON is retained alongside this report:

- `current-main.json`
- `issue-base.json`
- `pr-32621.json`
- `pr-33301.json`
- `pr-32630.json`
- `environment.txt`

## Limits and unfinished work

- This test uses synthetic probability tensors only; it does not load Kimi-K3 or any draft-model weights.
- It validates the renorm operations and the DSpark verify probability builder, not an end-to-end server request or throughput.
- Current `main` has callable renorm backends but `_DFLASH_SAMPLING_VERIFY_AVAILABLE` is false on HIP. The DFlash worker's non-greedy fallback was traced but not exercised with a full worker instance.
- No production code was changed. The report and validator are the only additions.
