# gfx942 causal_conv1d_update investigation

## Conclusion

Upstream issue `sgl-project/sglang#38622` is already addressed by upstream pull request `sgl-project/sglang#38623`, tested here at commit `dac135a1957f72122b492c31655d808e124c3de8`. The candidate passed the focused repository regressions and an independent token-by-token convolution reference on one AMD Instinct MI300X (`gfx942`). No code fix is duplicated in this report-only pull request.

The candidate's upstream base was `eb42598bddb146ac6c8e67cb63022e7420ed82ec`; this mirror's `main` was `0084030179bfba86bfeb6d43f7997d4076329d2c`. Because those bases differ, the candidate commit was checked out and tested directly rather than treated as a two-file diff against mirror `main`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one AMD Instinct MI300X, `gfx942`, Torch capability `(9, 4)`, UUID `36653834-3438-6562-3666-343964623164`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; ROCm/HIP `7.2.26015-fc0010cf6a`.
- Triton: `3.7.0+amd.rocm7.2.0.git89002410`.
- Installed SGLang source: `/sgl-workspace/sglang/python/sglang`, commit `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`, version `0.5.18.dev20260826+g937af8538b`.
- Installed native wheel: `/opt/venv/lib/python3.10/site-packages/sgl_kernel/common_ops.cpython-310-x86_64-linux-gnu.so`, version `0.4.6.post1`.
- Delivery checkout: `/job/sglang`, mirror `main` commit `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Job-private caches: `/tmp/sglang-cache-j-0325186515f5`; no model weights were downloaded.

## Installed-source first GPU baseline

The first GPU execution attempted the existing JIT test:

```bash
cd /sgl-workspace/sglang
/opt/venv/bin/python -m pytest \
  'test/registered/kernels/ops/mamba/test_causal_conv1d.py::test_causal_conv1d_update_matches_reference[True-exact-1-2-dtype0]' \
  -q
```

It failed after `23.1859` seconds, measured with `date +%s.%N` around the exact invocation. `hipcc` targeted `gfx942:sramecc+:xnack-` and failed because `cub/block/block_load.cuh` was unavailable. This is an installed-source baseline only and is not proof for later checkout changes.

The installed AOT control was also unsupported: `sgl_kernel.causal_conv1d_update` raised `AttributeError` because `torch.ops.sgl_kernel` had no `causal_conv1d_update` op, although the loaded native library was ROCm-linked.

The supported neighboring control was the installed Triton update. A three-case process took `7.19345` seconds (`date +%s.%N`), with `2.188863` seconds in `time.perf_counter` for imports and all cases. Against a stepwise `F.conv1d` reference:

- exact `state_len=3`, width 4: output max absolute difference `1.1920928955078125e-07`, state difference `0.0`, passed;
- oversized `state_len=8`, width 4: output max absolute difference `1.8087184429168701`, state max absolute difference `3.695260763168335`, failed;
- circular `state_len=8`, width 4, cursors `[0, 3]`: output max absolute difference `5.522923469543457`, state max absolute difference `2.936850070953369`, failed.

The full first-baseline artifact is `/job/baseline-first.json`.

## Mirror main behavior

On mirror `main` (`0084030179bf`), the same independent reference showed:

- exact tail state: passed;
- oversized tail state: output max absolute difference `2.320667266845703`, state max absolute difference `3.104637384414673`, failed;
- circular state: state max absolute difference `2.4806008333928223`, failed.

A deterministic circular control made the ignored cursor unambiguous:

- input tokens `[10, 20]`, state `[1, 2, 3, 4]`, width 3, cursor `2`;
- actual output `[13, 32]` and state `[10, 20, 3, 4]`;
- expected output `[17, 32]` and state `[1, 2, 10, 20]`.

## Candidate validation

The exact PR head was fetched from `https://github.com/My13ad/sglang.git` and checked out as `dac135a1957f72122b492c31655d808e124c3de8`.

Focused repository regressions:

```bash
cd /job/sglang
PYTHONPATH=/job/sglang/python \
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-0325186515f5/triton \
SGLANG_JIT_CACHE_DIR=/tmp/sglang-cache-j-0325186515f5/sglang-jit \
TORCHINDUCTOR_CACHE_DIR=/tmp/sglang-cache-j-0325186515f5/torchinductor \
XDG_CACHE_HOME=/tmp/sglang-cache-j-0325186515f5/xdg \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/opt/venv/bin/python -m pytest \
  test/registered/layers/mamba/test_causal_conv1d.py -q \
  -k 'test_causal_conv1d_update_uses_tail_of_oversized_state or test_causal_conv1d_update_supports_circular_state'
```

Result: `2 passed, 245 deselected` in `11.54` seconds; whole-process elapsed time was `19.1828` seconds.

The independent reference advanced one token at a time. Non-circular history used the trailing `width - 1` entries and appended each token while retaining the actual state length. Circular history used modulo cursor offsets, and each token was written with in-place `scatter_` at `cursor % state_len` before advancing the cursor. SiLU was applied after each `F.conv1d` output.

The whole corrected stepwise harness took `6.21539` seconds (`date +%s.%N`), with `1.339416` seconds in `time.perf_counter` for imports and all cases. All comparisons used float32 with `rtol=1e-4`, `atol=1e-4`.

| Case | Width | State length | Sequence length | Cursors | Output max abs diff | State max abs diff | Result |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| exact tail | 4 | 3 | 1 | none | `7.152557373046875e-07` | `0.0` | pass |
| oversized tail | 2 | 8 | 3 | none | `2.384185791015625e-07` | `0.0` | pass |
| circular issue | 3 | 4 | 2 | `[2]` | `2.384185791015625e-07` | `0.0` | pass |
| circular wraparound | 4 | 8 | 5 | `[0, 7]` | `2.384185791015625e-07` | `0.0` | pass |
| circular multi-wrap | 3 | 4 | 5 | `[2, 1]` | `4.76837158203125e-07` | `0.0` | pass |

Width 5 was explicitly rejected with:

```text
causal_conv1d only supports width between 2 and 4, got 5
```

## Architecture-specific limitations and scope

- The JIT C++ path is unavailable in this image because ROCm `hipcc` cannot find CUB headers while compiling for `gfx942`.
- The installed AOT `sgl_kernel` wheel does not register the causal-conv update op in `torch.ops.sgl_kernel`.
- The Triton candidate is supported on this `gfx942` stack and preserves both oversized tail storage and circular wraparound storage in the tested dense-update cases.
- The candidate intentionally supports widths 2 through 4 and rejects width 5.
- Circular `cache_seqlens` validation is checked for shape and `int32` dtype only when `validate_data=True`.
- Circular updates combined with speculative decoding/tree metadata were not exercised; this is an untested combination, not a claimed failure.
- No performance profiling or full-model weights were used. No upstream issue, pull request, or comment was posted or modified.
