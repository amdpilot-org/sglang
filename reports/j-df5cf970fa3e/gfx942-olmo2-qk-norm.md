# gfx942 OLMo-2 Q/K norm eager-capture-eager report

## Outcome

No code change is delivered in this PR. Mirror PR 271, commit
`8c273924a88d5b00420cebd6eae6d01daa8e8cba`, already contains the working
OLMo-2 eager-dispatch fix and a focused GPU regression. After finding that PR,
the local duplicate patch was discarded to avoid repeating a fulfilled scope.

This report preserves an independent gfx942 validation of upstream candidate
commit `a3faa0298cef6dff93a8f05b38e53944380f1fe5` (sgl-project/sglang PR
33416), which makes the same dispatcher change. Under the supported AITER
control, the candidate preserves fused Q/K dispatch across eager, capture, and
post-capture eager, and same-input Q/K outputs are bitwise identical.

## Environment

- Campaign: `repo-e2e-20260909`; job: `j-df5cf970fa3e`.
- GPU: one AMD Instinct MI300X, `gfx942`, node 5, GUID 6729, 206141652992 bytes VRAM.
- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`,
  operator-supplied local image ID
  `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Interpreter: `/opt/venv/bin/python`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP: `7.2.26015-fc0010cf6a`.
- Installed source: `/sgl-workspace/sglang`, commit
  `8eaffdf382e06b7dc50fc5c76cc5aad9c36bb0e4`.
- Delivery checkout base: `/job/sglang`, commit
  `0084030179bfba86bfeb6d43f7997d4076329d2c`.
- Native paths: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`,
  `/opt/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py`,
  `/sgl-workspace/aiter/aiter/__init__.py`,
  `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`, and
  `/sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so`.

## Method

The real `Olmo2Attention._apply_qk_norm` wrapper was exercised with reduced
synthetic `RMSNorm` modules. Q was bf16 `[17, 512]`, K was bf16 `[17, 128]`,
and both norm weights were bf16. The finite adversarial matrix included zero,
positive and negative values, `1e-3`, `1e-6`, `1e-12`, and `1e4` magnitudes.

The independent reference computed in float64:

```text
x * rsqrt(mean(x^2) + eps) * weight
```

then cast to bf16. The unchanged numerical gate was
`torch.allclose(output, reference, rtol=0.02, atol=0.02)`. Dispatch intent was
recorded with instance-level wrappers around the selected forward method and
the explicit native method. Timing used one synchronized `torch.cuda.Event`
around each eager/replay call and `time.perf_counter` around one capture.
Warmup was bounded to two calls; replay was bounded to one call.

## Installed-source baseline

The default installed-source run is not proof for later checkout changes. It
passed all six Q/K reference gates and same-input phase identity, but dispatch
was native/decomposed in eager and `forward_hip -> forward_native` during
capture because `vllm._custom_ops` is absent in this image.

- First GPU execution elapsed time: `0.43941682390868664 s`.
- Eager before capture: `0.38789400458335876 ms`.
- Capture wall time: `75.99375210702419 ms`.
- Replay: `0.13847899436950684 ms`.
- Eager after capture: `0.34202900528907776 ms`.
- Q/K phase identity: bitwise equal, max absolute difference `0.0`.

The meaningful supported neighboring control enabled the already-installed
AITER backend with `SGLANG_USE_AITER=1`; it did not replace the qualified
Torch/ROCm stack. It demonstrated the mismatch:

- Eager before capture: Q/K `forward_native`.
- Capture: Q/K `forward_aiter`.
- Eager after capture: Q/K `forward_native`.
- Q phase identity: not bitwise equal, max absolute difference `128.0`, mean `7.429541601854212`.
- K phase identity: not bitwise equal, max absolute difference `0.125`, mean `0.017110600191004136`.
- Both phase comparisons still passed the unchanged `rtol=0.02, atol=0.02` gate.
- First GPU execution elapsed time: `0.4411548860371113 s`.
- Eager before capture: `0.41708099842071533 ms`.
- Capture wall time: `92.00845565646887 ms`.
- Replay: `0.08760099858045578 ms`.
- Eager after capture: `0.28489598631858826 ms`.

## Candidate result

Upstream candidate `a3faa0298cef6dff93a8f05b38e53944380f1fe5` was tested with
`SGLANG_USE_AITER=1`:

- Eager before capture: Q/K `forward_aiter`.
- Capture: Q/K `forward_aiter`.
- Eager after capture: Q/K `forward_aiter`.
- Q and K eager-after versus capture-replay outputs were bitwise equal, with max absolute difference `0.0`.
- All six Q/K comparisons passed the unchanged reference gate.
- First GPU execution elapsed time: `0.4903939040377736 s`.
- First eager call: `17.770591735839844 ms` (includes first AITER selection work).
- Capture wall time: `73.79672024399042 ms`.
- Replay: `0.09080799669027328 ms`.
- Eager after capture: `0.3981980085372925 ms`.

The candidate therefore satisfies eager -> capture -> eager dispatch intent and
same-input Q/K output identity on gfx942. It is not delivered again because
mirror PR 271 already provides the equivalent working fix.

## Commands

```bash
rocm-smi --showproductname --showmeminfo vram

cd /tmp && /opt/venv/bin/python - < /job/baseline_first.py

cd /tmp && SGLANG_USE_AITER=1 /opt/venv/bin/python - < /job/baseline_first.py

cd /tmp && SGLANG_USE_AITER=1 PYTHONPATH=/job/sglang/python \
  /opt/venv/bin/python - < /job/baseline_first.py
```

The installed baseline is saved at `/job/baseline-first.json`, the AITER control
at `/job/baseline-first-aiter-control.json`, and the candidate result at
`/tmp/sglang-cache-j-df5cf970fa3e/candidate-a3faa029-aiter.json`.

## Boundaries and evidence consulted

- `vllm` is absent, so default `RMSNorm.forward_hip` falls back to native on this image.
- `torch.profiler` returned no CUDA events for `CUDAGraph.replay` on this ROCm stack; instance dispatch wrappers provide capture intent.
- No full model weights were downloaded, no toolchain was replaced, and no node-wide state was modified.
- Read-only context: sgl-project/sglang issue 33415 and PR 33416; amdpilot-org/sglang issues 213, 262, and 299; amdpilot-org/sglang PR 271.
- No upstream issue, PR, or comment was posted or changed.
