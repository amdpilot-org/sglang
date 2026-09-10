# MI300X structured diffusion block result

## Conclusion

The existing diffusion-side `ResBlock` convolution and `GroupNorm+SiLU` path passed the documented bf16 stability contract for all five structured input distributions. No numerical defect or regression was found, so this is a negative/coverage result rather than an operator fix.

The complete block used two 3x3 convolutions, the supported Triton fused `GroupNorm+SiLU` site, a native `GroupNorm`, and a residual activation. The independent reference used Torch convolution plus explicit fp32 mean/variance normalization and SiLU, with bf16 output casts.

## Environment

- Campaign: `repo-e2e-20260909`
- Mirror: `amdpilot-org/sglang`
- Branch: `amdpilot/j-55161f77e3a6`
- Base commit: `0084030179bfba86bfeb6d43f7997d4076329d2c`
- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`
- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch native path: `/opt/venv/lib/python3.10/site-packages/torch/lib`
- Relevant Torch native modules: `libtorch_hip.so`, `libaotriton_v2.so`
- Installed-source sglang path: `/sgl-workspace/sglang/python/sglang`
- Mirror-checkout sglang path: `/job/sglang/python/sglang`

## Installed-source baseline

The first successful GPU execution completed 70.362 seconds after task start, within the ten-minute requirement. It used the preinstalled source and the same five structured cases:

1. zeros
2. tiny finite values
3. mixed magnitudes
4. cancellation
5. skewed sparse values

The baseline used CUDA events with 30 calls per case and reported medians. It measured the complete block, a no-op contiguous conversion, and a channels-last conversion. Raw values are in `baseline-first.json`.

This installed-source baseline is evidence for the starting environment only. It is not proof for later checkout changes.

## Mirror-checkout result

The focused test added in this PR runs the same five structured distributions through the real `ResBlock` path and compares complete-block outputs to the independent fp32-statistics reference. It preserves the existing bf16 gate:

```python
torch.testing.assert_close(actual, reference, atol=7e-2, rtol=2e-2)
```

All five cases passed. The timing harness reran the affected cases from the mirror checkout and recorded:

- complete fused block: about 0.219-0.225 ms median
- independent reference: about 0.282-0.292 ms median
- no-op contiguous conversion: about 0.005-0.006 ms median
- NCHW to channels-last conversion: about 0.016-0.017 ms median
- channels-last to NCHW conversion: about 0.005-0.006 ms median
- channels-last two-pass `GroupNorm+SiLU` control: about 0.105-0.107 ms median

The two-pass control is a normalization-only measurement, not a complete-block measurement. Timings are medians on shared MI300X hardware and are not a performance guarantee.

Raw checkout results are in `results.json`.

## Reproduction

```bash
cd /job/sglang
export PYTHONPATH=/job/sglang/python
/opt/venv/bin/python -m pytest -q \
  python/sglang/multimodal_gen/test/unit/test_diffusion_conv_norm_structured_inputs.py
/opt/venv/bin/python reports/j-55161f77e3a6/run_structured_block.py \
  --output reports/j-55161f77e3a6/results.json
```

The test command completed with `5 passed`. The timing harness completed in about two seconds of GPU execution.

## Scope and limits

- Read-only roadmap context: upstream issue 23494 links merged PR 20319 (`fp8 MLA for diffusion model`) and merged PR 24651 (`fused all-reduce RMSNorm per-group quant`). Neither is the same structured conv/norm complete-block robustness case.
- One assigned MI300X GPU was used.
- No checkpoint or full model weights were downloaded.
- Total live synthetic allocations stayed well below 4 GB.
- Exactly five workload cases were used.
- The wall limit was 120 minutes; the investigation completed well within it.
- No generated-video quality test was run.
- No upstream issue, PR, or comment was posted or changed.
- Upstream issue 23494 is the AMD roadmap. Its current description and comments mention diffusion kernel fusion and AMD multimodal work, but do not contain a completed structured conv/norm robustness fix equivalent to this case.
- fp16 was not rerun in this bounded case set to keep the workload at five cases.
