# gfx942 equal-axis mRoPE investigation

## Scope and evidence

- Installed-source baseline: `reports/j-5dab44fcf888/baseline-first.json`.
- Mirror-main GPU matrix: `reports/j-5dab44fcf888/equal-axis-gpu-results.json`.
- Upstream reference: sgl-project/sglang issue 35345 and candidate commit `9b2e053ce0d203b368e68e915667295aea24df32`.
- Mirror `main` at `0084030179bfba86bfeb6d43f7997d4076329d2c` already contains the newer `[3, T]` mRoPE fused-kernel implementation and dispatch. This change does not duplicate that fulfilled scope.

## Result

On one AMD Instinct MI300X (`gfx942`), equal temporal/height/width rows reduced exactly to the documented one-axis control for four finite layouts:

- `[11, 11, 10]`, sectioned
- `[11, 11, 10]`, interleaved
- `[24, 20, 20]`, interleaved
- `[1, 1, 1]`, interleaved

The one-axis output also matched an independently derived PyTorch GemmaRMSNorm + NeoX RoPE + gate-deinterleave reference under the unchanged bfloat16 gate (`atol=rtol=2e-2`). A distinct-row adversarial control produced nonzero Q/K differences, confirming the equal-axis comparison is sensitive to per-axis positions.

The installed and mirror-main stock kernel initially failed on gfx942 because `_pdl_supported()` treated HIP compute major 9 as NVIDIA Hopper and emitted `griddepcontrol.launch_dependents`; AMD LLD rejected that instruction. The code change only guards this demonstrated architecture mismatch. The equal-axis matrix then ran without monkeypatching.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python /opt/venv/bin/python -m pytest -q \
  test/registered/kernels/ops/attention/test_fused_qk_rmsnorm_rope_gate.py
```

The bounded evidence scripts were run from `/tmp` with `/opt/venv/bin/python`; their commands, paths, numerical gates, raw metrics, and bounded timings are recorded in the two JSON files. No synthetic GPU burn, unbounded loop, sleep loop, full model weight, or environment replacement was used.

## Boundaries

PDL remains intentionally unsupported on HIP. The upstream candidate commit was consulted but not adopted because mirror `main` already contains a newer fulfilled mRoPE implementation. No upstream issue, PR, or comment was posted or changed.
