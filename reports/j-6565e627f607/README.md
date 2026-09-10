# Mask/attention GPU verification report

## Scope

- Upstream context: `sgl-project/sglang` issue `37553`.
- Tested candidate: `sgl-project/sglang` pull request `37120`, commit `f249808235a37dd7d071dffca531f06e1e7c4252`.
- Mirror base: `amdpilot-org/sglang` `main` at commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- This is a report-only verification. No source fix is duplicated or applied in this pull request.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, architecture `gfx942`, device capability `(9, 4)`.
- ROCm driver: `6.19.14.31400000`.
- Python: `/opt/venv/bin/python`, version `3.10.12`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- Installed `sglang` source: `/sgl-workspace/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Installed `sgl_kernel` package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel`.
- Native extension used by the installed package: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`.

## First GPU baseline

The installed-source baseline is recorded in `/job/baseline-first.json` and is explicitly **not** proof for later checkout changes.

- Command: `/opt/venv/bin/python -m pytest /sgl-workspace/sglang/test/registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_decode_attention -q`
- Working directory: `/sgl-workspace/sglang`
- Result: `1 passed`, `0 failed`.
- Reference comparison: the existing test compares Triton decode attention output against its pure-Torch reference.
- Timing method: wall-clock `date +%s%N` immediately before and after the pytest process, including Python startup, imports, Triton JIT, and test execution.
- First GPU execution elapsed time: `46.5` seconds overall; pytest reported `42.8` seconds.

## Independent mask and metadata harness

`verify_custom_mask.py` runs a finite set of varying draft trees and batch sizes on one GPU:

- Draft token counts: `4`, `8`, `16`.
- Growth batch sizes: `2`, `8`, `4`, `16`, `1`, `32`, `2`.
- Shrink batch sizes: `32`, `16`, `4`, `2`, `1`.
- Backends: `EagleVerifyInput.generate_attn_arg_prefill` and `DFlashVerifyInput.generate_attn_arg_prefill`.
- Total cases: `72` per checkout.

For every case the harness independently computes and compares:

- Expected `qo_indptr`.
- Expected `cum_kv_seq_len`.
- Expected `kv_indices`.
- Expected mask element count.
- Expected mask contents, including a deterministic mixed `True`/`False` pattern during shrink.

## Results

### Mirror `main` at `0084030`

- Command used the same harness with `PYTHONPATH=/job/sglang/python`.
- Result: `30` passed, `42` failed out of `72`.
- Growth phase: `20` passed, `22` failed.
- Shrink phase: `10` passed, `20` failed.
- All `qo_indptr`, `cum_kv_seq_len`, and `kv_indices` references matched.
- Failures were isolated to mask size and, during shrink, mask content.
- On batch shrink, `main` returned the full oversized stored mask instead of the exact current slice.

### Candidate `f2498082`

- Command used the same harness with `PYTHONPATH=/job/sglang-candidate/python`.
- Result: `72` passed, `0` failed out of `72`.
- Growth phase: `42` passed, `0` failed.
- Shrink phase: `30` passed, `0` failed.
- All mask, `qo_indptr`, `cum_kv_seq_len`, and `kv_indices` comparisons matched.
- The candidate’s own four GPU tests also passed: `4 passed`, `0 failed` in `17.75` seconds as reported by pytest.

## Timing

- Mirror `main` harness: `14.4` seconds wall clock.
- Candidate harness: `15.7` seconds wall clock.
- Candidate upstream test: `21.1` seconds wall clock, `17.75` seconds reported by pytest.

## Reproduction

```bash
git clone https://github.com/amdpilot-org/sglang.git /job/sglang
git -C /job/sglang fetch --depth 2 https://github.com/ItzDEXX/sglang.git f249808235a37dd7d071dffca531f06e1e7c4252
git -C /job/sglang worktree add --detach /job/sglang-candidate f249808235a37dd7d071dffca531f06e1e7c4252

PYTHONPATH=/job/sglang/python \
  /opt/venv/bin/python /job/sglang/reports/j-6565e627f607/verify_custom_mask.py \
  --label main-0084030 \
  --output /job/sglang/reports/j-6565e627f607/main-custom-mask.json

PYTHONPATH=/job/sglang-candidate/python \
  /opt/venv/bin/python /job/sglang/reports/j-6565e627f607/verify_custom_mask.py \
  --label candidate-f2498082 \
  --output /job/sglang/reports/j-6565e627f607/candidate-custom-mask.json
```

## Architecture-specific limitations

- Only one assigned MI300X (`gfx942`) GPU was used; no multi-GPU or multi-node behavior was tested.
- The qualified Torch/ROCm stack was preserved and no alternate framework or full model weights were downloaded.
- No CUDA graph capture or replay was exercised; the harness verifies the metadata and mask values returned by the affected prefill argument generator.
- No NPU or non-ROCm backend was tested.
- The candidate keeps storage at the observed peak rather than shrinking it on every batch decrease; this is consistent with the upstream PR’s stated “at most peak” behavior and avoids unbounded accumulation.
- The candidate commit is based on upstream `main` as of its parent commit `6afb5e17712e2e90b60ba8456ca893e529316869`; it is not a patch against the newer mirror `main` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
