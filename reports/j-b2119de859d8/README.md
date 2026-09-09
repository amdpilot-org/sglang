# MiniMax H3 ROCm denoising admission validation

## Result

The MiniMax H3 full-loop denoising platform guard rejected the assigned ROCm
GPU on current `main`. Adding `current_platform.is_hip()` admits ROCm while
preserving the explicit unsupported-platform error. A reduced deterministic
GPU denoiser using the real `minimax_h3_denoise_loop` matched an independent
Torch control on the same synthetic latents.

This is a single-GPU stage validation, not a full-model or multi-GPU E2E claim.

## Tested revisions

- Mirror `main`: `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- Upstream candidate: sgl-project/sglang pull request 35520, commit
  `315a5336a9383de31a61858d2ff26855b65512d6`
- The candidate PR is closed and unmerged. Its `is_hip()` admission change was
  applied on top of mirror `main` and adapted to the current guard, which also
  admits CPU, MPS, XPU, and Ascend NPU.
- Upstream `main` was queried during this run and still excluded ROCm.

## Guard and loop path

The guard is in `MiniMaxH3DenoisingStage._run_full_loop`. Current `main`
raised:

```text
MiniMax H3 full-loop denoise requires CPU, CUDA, MPS, XPU, or Ascend NPU
```

On the actual `RocmPlatform`, `current_platform.is_cuda()` was false and
`current_platform.is_hip()` / `is_rocm()` were true. With the fix, the guard
admits ROCm and an unsupported platform still raises:

```text
MiniMax H3 full-loop denoise requires CPU, CUDA, ROCm, MPS, XPU, or Ascend NPU
```

The full-loop stage dispatches to `MiniMaxH3DenoiseBranch`,
`minimax_h3_denoise_loop`, `prepare_timestep_plan`, `forward_kwargs`, and
`_minimax_h3_update_target_rows_`. The reduced harness exercises the real
branch construction, packed-row `index_copy_`, timestep planning, device
`index_select`, and the in-place Euler-eta0 update operations. It does not
claim coverage of the full H3 transformer or its attention backends.

## Reduced deterministic comparison

The harness is `reduced_denoiser.py`. It uses synthetic CPU-generated float32
latents with seed `0x5D3A11`, a packed t2va layout with 3 text rows, 8 video
rows of width 96, and 3 audio rows of width 32. It runs 3 denoising steps on
one MI300X and compares against an independent CPU Torch implementation that
does not call the SGLang loop.

Raw result:

```json
{
  "device_name": "AMD Instinct MI300X",
  "device_capability": [9, 4],
  "platform": "RocmPlatform",
  "steps": 3,
  "guard": {
    "rocm": "admitted",
    "unsupported_error": "MiniMax H3 full-loop denoise requires CPU, CUDA, ROCm, MPS, XPU, or Ascend NPU"
  },
  "video": {
    "max_abs": 4.76837158203125e-07,
    "mean_abs": 3.270906745456159e-08,
    "actual_sum": 81.80758666992188,
    "control_sum": 81.80758666992188
  },
  "audio": {
    "max_abs": 0.0,
    "mean_abs": 0.0,
    "actual_sum": -13.199492454528809,
    "control_sum": -13.199492454528809
  },
  "comparison_gate": {"rtol": 0.0, "atol": 2e-06, "result": "passed"}
}
```

The existing numerical gate in
`test_minimax_h3_denoise_loop.py::test_inplace_target_update_matches_scheduler_math`
was unchanged and continued to use `rtol=0, atol=0`. The full existing denoise
loop suite passed on both current and candidate source.

## Validation commands

Current source:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_minimax_h3_denoise_loop.py
```

Candidate source:

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_minimax_h3_admission.py \
  -k full_loop_denoise_admits_rocm
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_minimax_h3_denoise_loop.py
PYTHONPATH=/job/sglang/python /opt/venv/bin/python \
  reports/j-b2119de859d8/reduced_denoiser.py
```

The reduced run used job-private `HOME`, Torch, Triton, Inductor, and Hugging
Face cache paths under `/job/.cache/j-b2119de859d8`, with offline Hugging Face
flags. No model weights were downloaded.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, Torch capability `(9, 4)`.
- Raw GPU output: `logs/rocminfo.txt`, `logs/amd-smi.txt`, and
  `logs/rocm-smi.txt`.
- Python: `/opt/venv/bin/python`, Python 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Working source: `/job/sglang/python/sglang`.
- Installed source context initially imported from
  `/sgl-workspace/sglang/python/sglang`; it was not modified.
- Native modules observed: `/opt/venv/lib/python3.10/site-packages/torch` and
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel`; AIter loaded
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`.
- Required image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`,
  local image ID
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
  Docker/Podman/crictl were unavailable inside the job, so the local image ID
  could not be independently queried; it is recorded from the operator
  assignment.

## Explicitly outside this stage test

- Full MiniMax H3 model weights and checkpoint loading.
- Text encoder, video VAE, audio VAE, and output decoding.
- Full server request execution and generated media quality.
- Multi-GPU, tensor/sequence parallel, and the existing ROCm 2-GPU E2E case.
- H3 attention backend kernels, Cache-DiT, torch.compile, and performance.
- Benchmark or latency claims; none were measured.

## Limitations

- `ruff` was not installed in `/opt/venv`, so formatting and lint commands
  could not run. `py_compile`, `git diff --check`, and the focused test suites
  passed.
- The candidate commit was not present in the mirror, so its exact commit was
  not checked out. The tested change is the candidate's admission logic adapted
  to the newer platform list on mirror `main`.
