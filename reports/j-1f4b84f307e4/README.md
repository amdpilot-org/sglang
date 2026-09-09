# LongLive2 causal frame admission validation

## Result

Delivery PR: https://github.com/amdpilot-org/sglang/pull/85

Issue sgl-project/sglang 34367 is open, has no comments, and has no linked PR or
cross-reference events in its GitHub timeline. Current mirror `main` already
contains the block-aligned LongLive2 frame adjustment, so this investigation did
not duplicate a runtime fix. The nearby merged PR sgl-project/sglang 38226 only
quiets internal warmup frame searches; its head was also tested and passes.

The new synthetic GPU test validates both admission layers:

- `LongLive2T2VConfig.adjust_num_frames` preserves valid request frame counts.
- `_causal_block_count` rejects unadjusted request frame counts before denoising.
- `LongLive2CausalDenoisingStage.forward` rejects misaligned latent frame counts
  before the tiny GPU callback runs.
- Valid 8-, 16-, and 24-frame latent tensors preserve their full output shape.
- Invalid 7- and 9-frame latent tensors raise `ValueError` with no callback call.

No LongLive2 weights were downloaded, no server was started, no framework package
was changed, and no numerical gate or threshold was modified.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Operator-provided local image ID:
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942`, driver `6.19.14.314000`
- Python: `/opt/venv/bin/python`, Python 3.10.12
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch module: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native libraries: `/opt/venv/lib/python3.10/site-packages/torch/lib`
- ROCm libraries: `/opt/rocm-7.2.0/lib/libamdhip64.so`, `/opt/rocm-7.2.0/lib/librccl.so`
- Working source: `/job/sglang`
- Tested source module: `/job/sglang/python/sglang`
- Preinstalled import observed before `PYTHONPATH` override:
  `/sgl-workspace/sglang/python/sglang/__init__.py`
- Current mirror `main`: `ffe98a4279ba6e42d1f87dc4eeb6edb4887b9ea4`
- PR 38226 head: `6e7ff9f41cf251d150896b6998c2a1213309ef67`

## Synthetic validation

The test uses a tiny deterministic CUDA callback:

```python
denoised = chunk_latents.mul(2).add_(1)
torch.cuda.synchronize()
```

It records each callback's frame count, start frame, and output shape. The
callback is intentionally small and does not load model weights.

### Frame boundaries

| Case | Pixel frames | Latent frames | Blocks | Result |
| --- | ---: | ---: | ---: | --- |
| Minimum | 29 | 8 | 1 | shape preserved |
| Valid interior | 61 | 16 | 2 | shape preserved |
| Block edge | 93 | 24 | 3 | shape preserved |
| Off-by-one invalid | 65 or 66 | 17 direct request | invalid | request-level `ValueError` |
| Invalid low | direct latent | 7 | invalid | stage `ValueError`, callback not called |
| Invalid high | direct latent | 9 | invalid | stage `ValueError`, callback not called |

Normal sampling applies `adjust_num_frames` before `_causal_block_count`. The
65/66 request-frame failures therefore exercise the unadjusted request path;
they do not claim that a normal sampled request with 65 frames reaches the stage
unadjusted. Current config behavior rounds 65 to 61.

## Raw results

- Current `main`: `11 passed, 2 warnings in 0.95s`
  (`pytest-current-main.log`)
- PR 38226 head: `11 passed, 2 warnings in 0.90s`
  (`pytest-pr-38226.log`)
- Initial focused run before correcting a test-only block-shape expectation:
  `5 passed, 2 failed`; the runtime behavior was correct. The final focused file
  passes all 7 cases.

Warnings are pre-existing Cython `distutils.dep_util` deprecation warnings from
the environment, unrelated to this change.

## Reproduction

From `/job/sglang`:

```bash
export PYTHONPATH=/job/sglang/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_longlive2_pipeline_config.py \
  python/sglang/multimodal_gen/test/unit/test_longlive2_causal_frame_admission.py \
  -vv
```

The PR 38226 worktree used the same command with
`PYTHONPATH=/job/sglang-pr38226/python`.

## Limits and uncertainties


- This is synthetic stage/config validation, not a full LongLive2 server or model
  generation run.
- The request-level failure is validated at the request admission and stage
  boundary as a `ValueError`; no HTTP server was started.
- Issue 34367 has no issue-linked PR events. PR 38226 was tested as a nearby
  merged candidate, not represented as an issue-linked fix.
- No formatter binary was available in `/opt/venv`; the added test follows the
  surrounding pytest style and passes collection and execution.
