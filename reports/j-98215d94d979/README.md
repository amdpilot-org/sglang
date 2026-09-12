# Investigation of sglang#29160

Upstream issue: https://github.com/sgl-project/sglang/issues/29160

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2442

## Result

The prepared SGLang source already contains the available integration-level
solution through its FlashInfer dependency. No SGLang operator change is
justified.

The issue's later evidence identifies an out-of-bounds read in FlashInfer's
prebuilt SM100 routed dynamic-batch BMM cubin. The last CTA speculatively reads
one `int32` beyond `permuted_idx_to_token_idx`; allocator placement makes the
fault intermittent. FlashInfer PR 4237, cherry-picked to the release branch by
PR 4288, pads this route map by one element. PR 4288 reports deterministic
guard-page and compute-sanitizer validation. The kernel-side cubin defect and
workaround are outside SGLang's source tree.

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, SGLang pins
`flashinfer_python[cu13]==0.6.18` in `python/pyproject.toml` and uses the same
version for `flashinfer-cubin` and `flashinfer-jit-cache` in the CUDA installer.
The v0.6.18 tagged FlashInfer source contains the padding at the common launcher
allocation and at both independent routing metadata entry points. In contrast,
v0.6.12 allocates the route map without padding. The included audit checks this
failing-before/passing-after source distinction and rejects a partial fix.

Relevant dependency changes:

- https://github.com/flashinfer-ai/flashinfer/pull/4237
- https://github.com/flashinfer-ai/flashinfer/pull/4288
- https://github.com/flashinfer-ai/flashinfer/pull/3973 is a separate 2-CTA
  hang/cubin update and is not the issue-specific route-map fix.

## Reproduction and limitations

The reported runtime configuration requires eight NVIDIA B200 GPUs, CUDA 13,
GLM-5.2 NVFP4 weights, ModelOpt FP4, and FlashInfer's SM100 cubin. The assigned
machine has one AMD Instinct MI350X (`gfx950`) with ROCm 7.2, no FlashInfer
installation, and no GLM-5.2 weights. Consequently the illegal SM100 memory
access, full model, TP=8, speculative decoding, and serving workload were not
executed here. A tiny Llama serving fixture cannot qualify this architecture-
and-cubin-specific MoE failure, so it was intentionally not substituted.

The conclusion is source/dependency verification, not a claim that the original
production workload was reproduced on AMD hardware. A final runtime validation
would still require the reported B200-class configuration and should use a
guard-page probe or compute-sanitizer against the installed v0.6.18 cubin.

## Evidence retained outside the checkout

The following raw inputs are retained in the private job runtime directory
`/tmp/amdpilot-repo-j-98215d94d979`:

- `upstream_issue.json` and `mirror_issue.json`
- `flashinfer_pr4237.json`, `flashinfer_pr4288.json`, and
  `flashinfer_pr3973.json`
- `flashinfer-v0.6.12-trtllm_fused_moe_kernel_launcher.cu`
- `flashinfer-v0.6.18-trtllm_fused_moe_kernel_launcher.cu`
- `flashinfer_fix_to_v0618.json`

Run the audit after downloading those two tagged source files:

```bash
/tmp/amdpilot-repo-j-98215d94d979/venv/bin/python \
  reports/j-98215d94d979/audit_flashinfer_route_map.py \
  /tmp/amdpilot-repo-j-98215d94d979/flashinfer-v0.6.12-trtllm_fused_moe_kernel_launcher.cu \
  --expect vulnerable

/tmp/amdpilot-repo-j-98215d94d979/venv/bin/python \
  reports/j-98215d94d979/audit_flashinfer_route_map.py \
  /tmp/amdpilot-repo-j-98215d94d979/flashinfer-v0.6.18-trtllm_fused_moe_kernel_launcher.cu \
  --expect fixed
```
