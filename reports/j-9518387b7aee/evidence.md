# Independent review evidence

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2420 at exact commit `8929cfe77e5eb9ecc407ed9338da6b515d5376e4`.

Upstream issue: https://github.com/sgl-project/sglang/issues/31861

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2459

The prepared checkout was exactly the required recorded base, `358c163250ad3b1f62939b01ce1314a0a31a0365`, before revision switching. All raw logs and the candidate diff were preserved outside the checkout under `/job/review-evidence-j-9518387b7aee/`.

## Findings

The recorded base reproduced both relevant failures. A forced DeepSeek AMX branch returned a BF16 tensor and the CPU TopK wrapper forwarded caller-provided BF16 unchanged. Independently, FP32 values 1.001 and 1.002 both became 1.0 after a BF16 round trip.

At the exact candidate, its five focused tests passed. Independent checks showed that the modified DeepSeek gate bypasses packed BF16 output and returns FP32 for both BF16 and FP32 router weights. The Python CPU TopK boundary rejects BF16, FP16, and FP64 before dispatch and forwards FP32 unchanged.

The candidate is nevertheless incomplete against the original issue's model-wide contract. `python/sglang/srt/models/ernie4.py::MoEGate.forward` still calls `F.linear(hidden_states, self.weight)` without converting to FP32. With actual BF16 tensors it returned BF16, then the candidate's TopK guard raised `ValueError`. This is hardening plus a DeepSeek fix, not a complete guarantee that router logits are always FP32.

The optional specialized BF16-activation x FP32-weight AMX kernel remains absent. The DeepSeek FP32-weight fallback is semantically FP32 but unaccelerated.

## Architecture and import evidence

- CPU: AMD EPYC 9575F; Intel AMX is unavailable.
- GPU: one AMD Instinct MI355X, gfx950.
- Python source import: `/job/repo/python/sglang/__init__.py` at each checked-out revision.
- Native module: `/opt/venv/lib/python3.12/site-packages/sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg/sgl_kernel/__init__.py`.
- The installed native module did not expose `torch.ops.sgl_kernel.biased_grouped_topk_cpu` on this AMD host. Candidate native source was unchanged, so rebuilding native code was not applicable.
- The gfx950 `linear_bf16_fp32` reference returned FP32 and matched CPU FP32 matmul of identical BF16 operands with maximum absolute error `2.384185791015625e-07`.

## Raw evidence retained outside checkout

- `base-reproduction.txt`
- `candidate-regression.txt`
- `candidate-adversarial.txt`
- `candidate-remaining-counterexample.txt`
- `gpu-reference.txt`
- `candidate.patch`
- `candidate-pr.json`
- `upstream-issue.json`

Recommendation: `request_changes`. The candidate is a meaningful partial fix, but `fully_resolves_original` is false.
