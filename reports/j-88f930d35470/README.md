# DeepSeek-V4 gfx942 FlashMLA producer supplement

## Result

This supplement closes the missing evidence in [PR60](https://github.com/amdpilot-org/sglang/pull/60)
without changing its historical measurements. The earlier producer probe used
`FusedNormRopeKernel<bf16,128,64,...>::forward`, which selects
`fused_norm_rope_indexer`. That is the 128-FP8-plus-FP32-scale indexer layout,
not the head-dim-512 FlashMLA attention-K layout.

The new test calls the real production factory:

```python
_jit_compress_norm_rope_module(
    dtype=torch.bfloat16,
    head_dim=512,
    rope_dim=64,
    page_size=page_size,
    bf16_store=False,
)
```

The generated wrappers select
`FusedNormRopeKernel<bf16_t,512,64,64,false,0,false>::forward` and
`FusedNormRopeKernel<bf16_t,512,64,2,false,0,false>::forward`, both of which
dispatch to `fused_norm_rope_flashmla`. The test covers both source-derived
ordinary-worker pool shapes `(page_size, compression_ratio) = (64,4)` and
`(2,128)`. The logged `c128_page_size16` is not used by this constructor.

On one AMD Instinct MI300X (`gfx942:sramecc+:xnack-`):

- Control at exact commit `484c2286c993d36e862343c390a77439a003d244` failed
  the complete-cache comparison with 672 mismatched bytes for each pool shape.
- Candidate at that same exact commit plus only the supplied FP8 header patch
  passed both complete-cache comparisons with zero mismatched bytes.
- Candidate also passed every asymmetric-lane/swapped pack case, including
  signed zero, both sides of `±2^-11`, RNE ties, and the top segment.

Reporting correction: the retained historical JSON and logs report 1,344 because
`mismatch.numel()` counted both coordinates of each mismatch in the 2D cache.
Direct comparison of the saved actual/expected buffers gives 672 differing bytes
per control pool and zero per candidate pool. The driver now counts mismatch
rows; the historical raw evidence is unchanged.

This is single-GPU producer evidence only. It does not load a checkpoint, start
a server, or claim whole-model generation success. It authorizes preparation
of the later same-checkpoint TP8 A/B described by the failed strict generation
report in [PR59](https://github.com/amdpilot-org/sglang/pull/59), not acceptance
of that job. Issue context: [sgl-project/sglang#35122](https://github.com/sgl-project/sglang/issues/35122).

## Source and binary control

- Control worktree: detached at `484c2286c993d36e862343c390a77439a003d244`,
  clean before and after probing.
- Candidate worktree: same detached commit plus only the supplied production
  header patch; no merge or rebase.
- Delivery branch: based exactly on PR60 head
  `0efcbf6a12c481b35e35fa3e15e8916e52499f8c`.
- Baseline header SHA256:
  `6b686494e4ee7ac6f12f972e1033837ff020d1629e833e06f57f304ed0c5f022`.
- Patched header SHA256:
  `16ee2f59330081e0015390d4f895cbd7aa093b55ace86723a668699b130ff375`.

Each arm used a separate private JIT cache and a fresh Python process with
`PYTHONPATH` pointing only at that runtime source. The normal production JIT
emitted:

```text
-fPIC -D__HIP_PLATFORM_AMD__=1 -fno-gpu-rdc
--offload-arch=gfx942:sramecc+:xnack-
-DUSE_ROCM -std=c++20 -O3 -DHIP_FP8_TYPE_FNUZ=1
```

The loaded native producer binaries and their SHA256 values were:

- Control page 64:
  `2bf6aec1c306d578e45a09a8eccd515adf6166b5060d5cc86daf2e5e58d3b385`
- Control page 2:
  `787797de6292589226253acc71379aa929492b78b86dd06ecf3f45c7c577611f`
- Candidate page 64:
  `3dcd9392f9fd52b0d623ebc338ccd125b5aeef05d94e632dd95b8ef8ea137fdf`
- Candidate page 2:
  `4f54e9b75b35fb662a332e8a60c8ba55e1a5a92bf29ed8e5ddf766866099b229`

The generated `build.ninja` and wrapper source, source/header hashes, loaded
`.so` paths, SHA256 values, complete actual/expected cache bytes, and pack case
bytes are retained under this report directory and in the collected `/job`
artifact tree.

## Producer fixture

The fixture uses three BF16 input rows `[ones(512), -ones(512), ones(512)]` with
`eps=0.0`, giving a nonzero norm and exactly unit RMS. The first 448 gamma
elements contain seven 64-element groups. Each group repeats
`[64,128,160,224,-64,-128,-160,-224]` eight times, scaled by `2^-group`. The
last 64 rope gamma elements repeat `[1,-2,0.5,-0.75]` sixteen times.

`freqs_cis` is contiguous FP32 with shape `(ratio+1,64)` and interleaved
`[1,0]` identity rotation. The decode plan has sequence lengths
`[ratio,2*ratio,ratio+1]`; the third row is skipped by decode-boundary gating.
`out_loc` is `[P-1,P+1,2P]`.

Each physical page is `((584*P+575)//576)*576` bytes and begins filled with
sentinel `0xA5`. The expected cache writes only two payloads, two 128-byte rope
regions, and seven UE8M0 scale bytes (`127..121`). The eighth scale byte,
neighboring slots, skipped third row, and all page padding remain sentinels.
The comparison covers the entire cache.

Expected non-rope bytes come from the unscaled per-group pattern cast by Torch
to `float8_e4m3fnuz`, not from `pack_fp8`. Actual FNUZ bytes are also decoded and
multiplied by `2^-group` before an exact comparison with `±gamma[:448]`. Rope
bytes are compared exactly against `±gamma[448:]` cast to BF16.

## Pack coverage

The supplemental pack test uses the PR60 production-header wrapper, which
includes the runtime `fp8_utils.cuh` and calls the real `pack_fp8` in both lanes.
It covers asymmetric pairs and their swapped versions, preserves signed-zero bit
patterns without deduplication, and checks both FP32 `nextafter` neighbors on
both sides of `+2^-11` and `-2^-11`. The negative neighbor is used directly; it
is not negated a second time.

The existing policy clamp remains `224`. The observed finite `240 -> 224` change
is therefore a consequence of corrected FNUZ encoding after the existing input
clip, not a new clipping policy. The software FN fallback is affected by the
rounding fix. The `gfx950` hardware conversion branch is unchanged and was not
run.

## Initial fixture failures retained

The first attempts exposed fixture/evidence issues, not production defects. All
raw failures are retained under `initial_failures/`:

1. The 64-element gamma group was not repeated.
2. The expected non-rope payload used one 64-byte group instead of seven.
3. BF16 raw serialization needed an explicit `uint16` bit view.
4. The loaded `.so` matcher confused rope-dim 64 with page size 64.
5. The candidate expected cache reused `+gamma` for the negative row.

After correcting those fixture/evidence bugs, the production fix and expected
math were not changed to make either arm pass.

## Reproduction

From a ROCm PyTorch environment on one gfx942 GPU:

```bash
PYTHONPATH=python SGLANG_JIT_CACHE_DIR=/tmp/sglang-jit \
python -m pytest \
  python/sglang/test/kernels/deepseek_v4/test_fp8_pack.py \
  python/sglang/test/kernels/deepseek_v4/test_compress_norm_rope.py
```

The collected A/B driver is retained as `run_single_gpu_ab.py`. It requires an
arm name, an explicit runtime source root, the PR60 probe source, and a private
`SGLANG_JIT_CACHE_DIR`. It records the generated build files, loaded `.so`
paths/hashes, complete cache bytes, and pack case bytes.

## Limitations

- Only one assigned MI300X gfx942 GPU was used.
- No model checkpoint was loaded or downloaded.
- No multi-GPU server or TP8 job was launched.
- No gfx950, gfx1200, gfx1201, or CUDA device was run.
- The source-derived pool shapes were not corroborated against a live TP8
  runtime in this job; the later TP8 A/B must resolve and record those attrs.
- Passing this producer test does not reverse the failed strict generation
  result in PR59 and does not establish whole-model generation correctness.
