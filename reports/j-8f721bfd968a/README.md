# MI300X fused-MoE output reuse experiment

This is a bounded follow-up to amdpilot-org/sglang issue 229 and read-only context
from sgl-project/sglang issue 32312. It investigates the full unquantized BF16 Triton
fused-MoE path's fresh output allocation versus its documented in-place buffer reuse
across distinct 1-, 4-, and 16-token batches. It does not repeat issue 229's original
wave-level bug trigger and does not post to or modify either upstream issue.

## Result

The installed-source baseline is recorded outside the repository at
`/job/baseline-first.json`. Its first gfx942 execution took 0.972 seconds and both
fresh and reused outputs matched an independent per-expert PyTorch reference.

The persistent checkout experiment in `results.json` used one AMD Instinct MI300X
(`gfx942`), the qualified Torch/ROCm stack, and the installed `sgl_kernel` native
activation module. Fresh outputs were distinct from their inputs and unchanged
inputs; reused outputs retained one stable address, preserved the input/output dtype
and shape contract, and left 32-element before/after guard regions unchanged.
All six numerical cases passed `torch.allclose` with `rtol=2e-2` and `atol=2e-2`.
The bounded timing matrix used two warmups and ten timed calls per case; mean
latencies ranged from 0.256 to 0.300 milliseconds.

The checkout also makes non-symmetric fresh allocation avoid the unnecessary
symmetric-memory context. This preserves the existing symmetric path while allowing
the disabled path to allocate normally. Unsupported float64 and
`inplace + no_combine` variants fail explicitly rather than being forced through.

## Reproduction

```bash
PYTHONPATH=/job/sglang/python \
TRITON_CACHE_DIR=/tmp/sglang-cache-j-8f721bfd968a/triton \
/opt/venv/bin/python reports/j-8f721bfd968a/run_reuse_experiment.py

PYTHONPATH=/job/sglang/python \
/opt/venv/bin/python -m pytest -q \
test/registered/unit/layers/moe/test_fused_moe_output_reuse.py
```

The experiment uses no model weights, no toolchain replacement, and no unbounded
GPU work. Build caches are job-private under `/tmp/sglang-cache-j-8f721bfd968a`.
