# gfx942 top-k v2 threshold-overflow investigation

## Scope

This report covers sgl-project/sglang issue 35257 on one assigned AMD Instinct
MI300X (gfx942). It deliberately separates the tied-threshold and
unique-value overflow case from the earlier continuous-score overflow case.

The working candidate is sgl-project/sglang pull request 37941, commit
`7a83d7fdd548caf1ed4da9525e1a17724661172c`. Because that upstream change
already fixes the register and streaming paths, this branch intentionally
contains no duplicate kernel change.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: `gfx942`, UUID `GPU-b5c590cf4c10631d`, AMD Instinct MI300X
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Candidate source: `/job/sglang` at `7a83d7fdd548caf1ed4da9525e1a17724661172c`
- Candidate JIT native module:
  `/tmp/sglang-cache-j-8014dfc3a10a/jit-candidate-verify/gfx942/sgl_kernel_jit_dpsk_v4_topk_v2/build-2bce6de0bd044983/deps-dbec51fef94405e4/sgl_kernel_jit_dpsk_v4_topk_v2.so`

## Installed-source baseline

The preinstalled source was `/sgl-workspace/sglang` at
`8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4` with local modifications. Its
top-k v2 JIT build failed on gfx942 because the ROCm compile path could not
find `cooperative_groups.h`. The first bounded GPU execution attempt took
11.013 seconds and produced no kernel output. This is an installed-source
baseline only and is not evidence about later checkout changes.

As the supported neighboring control, `torch.topk` on an `8 x 262144` fp32
input selected exactly the expected score multiset and threshold bucket. Ten
synchronized calls took 0.002132 seconds, or 0.213 ms per call.

## Results

On mirror `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c`:

- 2,047 unique values in one fp16 coarse bin: passed
- 2,048 unique values in one fp16 coarse bin: passed
- 2,049 unique values in one fp16 coarse bin: failed, 1 true top-k member missing
- 3,000 unique values in one fp16 coarse bin: failed, 952 true top-k members missing

On PR 37941 commit `7a83d7fdd548caf1ed4da9525e1a17724661172c`:

- The eight focused single-coarse-bin tests passed in 24.831 seconds.
- The complete affected test file passed: 286/286 in 35.744 seconds.
- An independent oracle passed tied register and streaming cases with 1,000
  scores above the threshold and 3,000 bit-identical scores at the threshold.
- The same independent oracle passed unique-value counts of 2,047, 2,048,
  2,049, and 3,000 in one fp16 coarse bin.

The independent oracle accepts any subset of bit-identical values at the
threshold, but requires the selected score multiset, all greater-than-threshold
indices, and threshold-bucket accounting to be exact. For unique values it
also requires the exact expected index set.

## Reproduction

Use one gfx942 GPU and keep the JIT cache outside the checkout:

```bash
cd /job/sglang
git switch --detach candidate-pr-37941
PYTHONPATH=/job/sglang/python \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-8014dfc3a10a/jit \
PYTEST_ADDOPTS='-p no:cacheprovider' \
timeout 600 /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/attention/test_topk_v2.py
```

The independent boundary oracle is recorded in `results.json`. It uses one
plan, transform, and synchronize execution per case and `time.perf_counter`
for timing.

## Architecture-specific limitations

- The tested gfx942 paths are the register and streaming implementations.
- PR 37941 documents that the CUDA-only `TopKCluster` path still truncates an
  overflowing threshold bin. That path is not compiled on this ROCm target
  and was not exercised here.
- The preinstalled source's `cooperative_groups.h` failure is specific to
  that dirty installed checkout and the preserved ROCm 7.2 stack; the clean
  mirror `main` and candidate checkouts compiled and ran.
- No model weights were downloaded and no node-wide state was modified.
