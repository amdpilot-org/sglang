# ROCm FP4 expert dequantization TP8 validation

## Source

- Issue: `sgl-project/sglang#35122`
- Source PR: `amdpilot-org/sglang#56`
- Requested source commit: `710dc165936d617826c492016ed9189875376bdc`
- Branch base (`amdpilot-org/sglang` `main`): `db272201a2dbd72e5699e443240a851f1313ad45`
- The six PR56 commits were cherry-picked onto the required branch. The final branch commit at validation time was `082ad8ce15176ac80fa0afd4daa3a3ef71bbb126`; commit hashes differ after cherry-picking, but the base-to-head diff matches PR56 at the requested commit.
- The image-installed `/sgl-workspace/sglang` checkout was `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`. Full-model validation did not use that source directly; it set `PYTHONPATH=/tmp/sglang-j-a486ba1b7dc4-src/python` to the patched checkout.
- No PR32333 backend-selection changes are included.

## Runtime

- Hostname/container: `banff-cyxtera-cx57-4`; Ubuntu 22.04.5 LTS.
- The container runtime did not expose a Docker image name or digest. The runtime stack was ROCm 7.2.0, HIP 7.2.26015-fc0010cf6a, Torch `2.9.1+rocm7.2.0.git7e1940d4`, Python 3.10.12, Transformers 5.12.1, Triton `3.7.0+amd.rocm7.2.0.git89002410`, and AITER from `/sgl-workspace/aiter`.
- Eight visible GPUs were AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 CUs each. Serials and Torch UUIDs are recorded in `/job/gpu_identities.json` and `/job/torch_hip.json`.
- This was MI300X/gfx942, not MI308X. The upstream 200-question score was not rerun.

## Checkpoint

- Read-only mount: `/models/DeepSeek-V4-Flash-0731`
- Identity: `deepseek-ai/DeepSeek-V4-Flash-0731`, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Index SHA256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Parsed index: 72,317 tensor entries referencing exactly 48 shards.
- Config advertises `quant_method=fp8` while routed experts use `expert_dtype=fp4` (packed I8 with F8_E8M0 scales).
- No model files were downloaded or modified.

## Synthetic regression

Patched command:

```bash
cd /tmp/sglang-j-a486ba1b7dc4-src
export PYTHONPATH="$PWD/python"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
pytest -q test/registered/unit/layers/quantization/test_fp8_rocm_fp4_dequant.py -vv
```

Result: **5 passed** in 37.52s. No `AITER_CONFIG_*` workaround was set. The tests cover:

- three ROCm FP4-to-FP8 expert shapes;
- `float8_e4m3fnuz` weights, `float32` block scales, exact weight/scale bytes, AITER shuffled layout, and reconstruction against the FP4 reference;
- native FP4/default behavior remaining unshuffled FP4;
- non-ROCm conversion to `float8_e4m3fn` with exact expected weights and scales.

Baseline command used a detached worktree at `db272201a2dbd72e5699e443240a851f1313ad45` with the same test file. Result: **3 failed, 2 passed**. All three ROCm dequant cases failed because weights remained `torch.float4_e2m1fn_x2`; native-FP4/default and non-ROCm controls passed. Full output is in `/job/pr56_baseline_regression.log`.

## Full-model TP8 control

Server wrapper:

```bash
/job/run_tp8_server.sh
```

Explicit environment:

```text
SGLANG_USE_AITER=1
SGLANG_DSV4_FP4_DEQUANT=1
SGLANG_HACK_FLASHMLA_BACKEND=triton
PYTHONPATH=/tmp/sglang-j-a486ba1b7dc4-src/python
```

Server flags:

```text
--host 127.0.0.1
--port 31322
--model-path /models/DeepSeek-V4-Flash-0731
--tp 8
--cuda-graph-max-bs-decode 8
```

The first full-model attempt used the default TileLang FlashMLA backend. All eight ranks loaded and dequantized FP4 experts, but full decode capture failed in `dpsk_v4_fp8_partial_kernel` during TileLang lowering with:

```text
ValueError: Check failed: src_info.order < dst_info.order (6 vs. 4)
```

The complete failure is preserved in `/job/server_full_graph_tilelang_failure.log`. The retry only set `SGLANG_HACK_FLASHMLA_BACKEND=triton`; it made no source or dependency changes and preserved the installed Torch/ROCm ABI.

The successful run loaded all 48 original shards, logged `Dequantized FP4 expert weights to FP8.` on ranks 0–7, and completed full decode CUDA graph capture for ranks 0–7 with batch sizes `[1, 2, 4, 8]`. `/job/graph_capture_summary.json` records all eight start/end ranks and `/job/server.log` contains the full server log.

## Inference

Validation command:

```bash
python /job/validate_tp8.py
```

Each case sent two greedy requests with `temperature=0.0`, `top_p=1.0`, and `max_new_tokens=2`. The declared sample passed:

- `12 + 34` → exact output `46` on both runs.
- `7 * 8` → exact output `56` on both runs.
- `100 - 45` → exact output `55` on both runs.
- Capital of France → coherent output containing `Paris` on both runs.

Raw outputs and pass status are in `/job/tp8_validation.json` and `/job/tp8_validation.log`.

## Limitations

- This is not a 200-question GSM8K run and does not reproduce the upstream score.
- This worker is MI300X/gfx942, not MI308X.
- The default TileLang FlashMLA path fails during graph capture on this runtime; the successful control explicitly selected the Triton backend.
- Longer greedy continuations can diverge after the expected answer. `/job/tp8_validation_longer_nondeterminism.log` preserves that observation. The declared deterministic sample therefore checks exact two-token numerical answers and a stable factual answer.
- The container did not expose its Docker image name or digest; the actual runtime versions and image-installed source commit are recorded above.
