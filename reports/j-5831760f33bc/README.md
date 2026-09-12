# Independent review of PR 2416

Candidate: https://github.com/amdpilot-org/sglang/pull/2416 at `0dbcbe1fea5b4d098832cd47dc5e0633e9e8b963`

Upstream issue: https://github.com/sgl-project/sglang/issues/30887

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2344

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2451

## Verdict

Recommendation: **accept**. The candidate is a narrow Python-only correction to the failing post-load path. On the recorded base, its issue-specific regression reaches the real `ModelOptNvFp4FusedMoEMethod.process_weights_after_loading` implementation, emits the reported `w2_weight_scale` shape `(1, 2816, 22)`, and fails at the original gated-padding assertion. At the exact candidate commit, the same regression passes. The focused candidate suite reports 17 passed tests and 38 passed subtests.

An independent oracle checked logical gate/up markers, zero-filled padding, packed `w2` columns, block-scale columns, aligned no-op identity, and invalid-layout rejection for per-rank intermediate sizes 16, 32, 48, 64, 80, 112, 128, 352, and 704. The reported 352-channel case also passed with tensors resident on the assigned AMD Instinct MI350X/gfx950 GPU.

The implementation pads each gated `w13` half independently to a multiple of 64 before block-scale swizzling. It applies the corresponding `w2` packed-column and scale-column padding, preserving the gate/up boundary. No native C++, FlyDSL, or extension source changed, so a native rebuild was not applicable. Import inspection confirmed that `sglang` and `modelopt_quant.py` came from `/job/repo/python`, while Torch came from the prepared ROCm environment.

## Scope and limitations

`fully_resolves_original` is recorded as false because this environment cannot execute the original end-to-end contract: it has one AMD gfx950 GPU, not two NVIDIA B300 GPUs, and the `nvidia/Gemma-4-26B-A4B-NVFP4` weights were unavailable. Therefore this review does not independently verify two-rank CUDA startup, the real FlashInfer CUTLASS block-scale kernel, full model loading, generation, or semantic parity. There is no observed tensor-level counterexample to the candidate; the remaining gap is architecture/model-level verification.

The candidate's own prose mentions external two-B300 serving results, but this verdict does not treat that prose as proof. It rests on the independently reproduced assertion, source-path verification, the focused regression, code inspection, and independent CPU/GPU tensor invariants.

## Commands and evidence

Raw command output was preserved outside revision switching under `/job/review-evidence-j-5831760f33bc/`.

```bash
# Recorded base: expected failure, exit 1
/tmp/amdpilot-repo-j-5831760f33bc/venv/bin/python -m pytest -xvv \
  test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_padding.py::TestNvfp4MoeIntermediatePadding::test_process_weights_after_loading_pads_gemma4_tp2_before_swizzle

# Exact candidate: exit 0
/tmp/amdpilot-repo-j-5831760f33bc/venv/bin/python -m pytest -q \
  test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_padding.py \
  test/registered/unit/layers/quantization/test_modelopt_nvfp4_moe_scales.py

# Independent boundary and GPU-resident tensor oracle: exit 0
/tmp/amdpilot-repo-j-5831760f33bc/venv/bin/python \
  /job/review-evidence-j-5831760f33bc/candidate/adversarial.py
```

Key baseline evidence:

```text
WARNING ... NVFP4 w2_weight_scale K' not multiple of 4: shape=(1, 2816, 22), group_size=16
AssertionError: The intermediate size required padding, but padding is also implemented for gated activations
```

Key candidate evidence:

```text
17 passed, 17 warnings, 38 subtests passed
cuda 352 [(2, 768, 16), (2, 768, 2), (2, 32, 192), (2, 32, 24)] ok
gpu AMD Instinct MI350X 7.2.26015
invalid_scale_rows rejected
invalid_w2_width rejected
```
