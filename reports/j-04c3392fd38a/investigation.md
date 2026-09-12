# Compact target-verify CUDA graph correction review

Upstream issue: https://github.com/sgl-project/sglang/issues/36481

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1352

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1237 at `970ab777f74aea05733c8f5444642f26006c3f59`

Independent review PR: https://github.com/amdpilot-org/sglang/pull/1302

## Result

The review's counterexamples are correct. `TritonAttnBackend` does not opt in to
ragged verify graphs, and `HybridLinearAttnBackend` requires both its full and
linear children to opt in. The candidate therefore skips compact target-verify
capture and the existing replay admission check selects eager execution. It
does not make Triton `extend_attention_fwd` safe for a compact CUDA graph.

The candidate's guard is nevertheless a valid narrow crash-avoidance fix. On
the prepared base, removing it makes the focused regression enter `warmup()`
despite the backend declaring ragged graphs unsupported. Restoring it skips
capture, and the strengthened regression also verifies that replay admission is
false. Supported compact backends and non-compact capture still continue.

No source change enabling Triton ragged graphs is justified here. The available
device is AMD Instinct MI350X/gfx950 under ROCm 7.2, while the reported failure
requires NVIDIA B300/CUDA 13.2 and unavailable Qwen hybrid plus DSpark weights.
The original illegal memory access at batch size 20 and eight tokens per request
was not reproduced, and `extend_attention_fwd` was not executed by this test.

## Evidence

- `evidence/failing-before.log`: guard removed; 5 tests passed and the
  unsupported compact capture test errored after entering poisoned `warmup()`.
- `evidence/passing-after.log`: guard restored; all 6 tests passed.
- `evidence/environment.log`: one MI350X/gfx950, Torch ROCm 7.2, no CUDA runtime.

The retained source correction is intentionally limited to preventing capture
of a graph the backend explicitly declares unsupported. Compact target-verify
CUDA-graph support for Triton-backed Qwen hybrid models remains unresolved and
requires validation on the reported architecture and model workload.
