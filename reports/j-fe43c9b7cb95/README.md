# Investigation report: diffusion LoRA with online FP8

The prepared base commit already contains the quantization-aware correction
proposed for the open issue. No runtime source change was justified.

`BaseLayerWithLoRA.can_merge_base_weight` admits ordinary unquantized linear
methods and rejects quantized `LinearBase` methods. The pipeline uses that
capability to make `auto` select dynamic LoRA and to reject explicit `merge`
before applying an adapter. This matches the implementation in open upstream
PR https://github.com/sgl-project/sglang/pull/35975.

The retained GPU fixture uses the actual online `Fp8Config()` post-load path on
one assigned MI350X (gfx950). It demonstrates the physical transpose from a
logical `(48, 32)` weight to `(32, 48)` float8 runtime storage, invokes the
legacy merge routine to preserve the failing behavior, verifies the current
policy, and compares dynamic LoRA output with an independently expressed BF16
matrix-chain reference. It also checks ordinary and mixed-layer boundaries.

Run from the repository root:

```bash
export PYTHONPATH=/job/repo/python
export HIP_VISIBLE_DEVICES=0
/tmp/amdpilot-repo-j-fe43c9b7cb95/venv/bin/python \
  reports/j-fe43c9b7cb95/reproduce_fp8_lora_policy.py
```

The exact MiniMax-H3 FL2VA TP2 startup and generation were not reproduced:
the assigned environment has one AMD GPU and does not contain the reported
model or LoRA weights. This fixture qualifies the merge-policy mechanism and
GPU execution only, not that model architecture, TP2, CUDA behavior, video
semantics, or performance.
