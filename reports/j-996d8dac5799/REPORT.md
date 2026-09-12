# GLM 5.3 MI300 scheduler hang investigation

## Outcome

The original failure was not reproduced, and no SGLang product-code change is
justified by the available evidence. The report requires a long-running GLM 5.3
deployment on eight MI300X (`gfx942`) GPUs and appears after roughly one million
tokens. This job exposes one MI350X (`gfx950`) GPU and has no GLM 5.3 weights.

The stack trace points at the scheduler only because it synchronizes a failed GPU
operation. Immediately before the hang, the issue log repeatedly identifies an
AITER block-scaled FP8 fused-MoE kernel for this tuple:

```text
('gfx942', 304, 2048, 6144, 256, 257, 9, ..., per_1x128, True, False)
_ZN5aiter46fmoe_bf16_blockscaleFp8_g1u1_vs_ps_silu_64x256E
```

SGLang delegates that operation directly to `aiter.fused_moe.fused_moe` in
`python/sglang/srt/layers/moe/moe_runner/aiter.py`; it does not launch or index the
ASM kernel itself.

## Related fixes checked

- SGLang PR https://github.com/sgl-project/sglang/pull/37769 is not a fix for this
  report. It prevents an immediate assertion in the *Triton* MoE fallback for
  clamped SwiGLU. The issue log explicitly uses the AITER runner after sustained
  serving.
- AITER mirror PR https://github.com/amdpilot-org/aiter/pull/81 tested the reported
  `M=304, N=6144, K=2048` boundary and adjacent controls on one real MI300X. On its
  tested AITER main commit the dispatcher selected a one-stage default kernel and
  all processes exited normally. It reports this as an already-fixed/negative
  crash-boundary result, not as a full GLM serving reproduction.
- AITER mirror PR https://github.com/amdpilot-org/aiter/pull/20 independently tested
  `M=288/304/336/352`, eager and HIP graph replay, on MI300X. All 24 cases passed
  against its dequantized Torch reference and selected a one-stage kernel.

Those AITER results explain why duplicating a speculative scheduler guard in SGLang
would be inappropriate. They do not prove that the reporter's older AITER kernel,
eight-GPU topology, or million-token workload is fixed.

## Local GPU evidence

Hardware was measured with `rocm-smi`: one AMD Instinct MI350X, `gfx950`, 270566162432
bytes VRAM. The prepared interpreter reports Torch `2.11.0+rocm7.2`, HIP
`7.2.26015`, and one visible GPU.

The installed AITER test was run for the exact available tensor geometry and two
independent token-count controls:

```bash
timeout 600 env CUDA_VISIBLE_DEVICES=0 AITER_LOG_LEVEL=INFO \
  AITER_CONFIG_FMOE=/sgl-workspace/aiter/aiter/configs/tuned_fmoe.csv \
  PYTHONPATH=/sgl-workspace/aiter \
  /tmp/amdpilot-repo-j-996d8dac5799/venv/bin/python \
  /sgl-workspace/aiter/op_tests/test_moe_2stage.py \
  -q 5 -t M -dim 2048,6144 -e 256 -k 9 -a silu -s f -p t \
  -hip 0,0 --no-flydsl-csv
```

For `M=288`, `304`, and `320`, every process exited 0, selected
`run_1stage=True`, padded to tuning token 512, and loaded the gfx950
`..._1tg_ps_32x384` HSACO. The exact `M=304` run reported aggregate
`logits_diff=0.000171621`. The test's legacy elementwise `atol=rtol=0.01` diagnostic
also printed `failed!` for large-magnitude BF16 outputs; this is retained in the raw
log and is not represented as an accuracy pass. No process hung or raised a GPU
memory fault.

Raw logs and exit codes are under `reports/j-996d8dac5799/raw/`.

## Limitations

- No MI300X/gfx942 GPU was assigned, so the reported ISA/kernel was not executed
  locally.
- Only one GPU was available; TP8, cross-rank behavior, and the reported scheduler
  topology were not exercised.
- GLM 5.3 weights were unavailable, and the million-token time-to-failure workload
  was not run. A tiny Llama fixture would not qualify the model-specific MoE path,
  so it was intentionally not substituted.
- The local synthetic run validates one AITER kernel geometry on gfx950 only. It is
  not a model, semantic-accuracy, longevity, or distributed-serving reproduction.
- AITER is an external runtime dependency here. If the old gfx942 `64x256` kernel is
  still faulty in the reporter's pinned build, the correction belongs in AITER or
  its dispatch/configuration, and requires reproduction on that architecture.

