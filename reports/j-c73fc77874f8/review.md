# Independent review of candidate 8574da7c83c3247d0029e5a63aeaf8dcc6521970

Upstream issue: https://github.com/sgl-project/sglang/issues/32669

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1990

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2072

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2032

## Recommendation

Accept. The candidate is a narrow, source-level correctness gate and fully resolves the original issue's required behavior by preventing the unsafe non-EP DeepSeek-V4 TBO strategy whenever attention TP is greater than one. It preserves non-EP attention-TP1 and EP TBO. No remaining source-level counterexample was found within the original contract.

## Evidence

The image-prepared checkout was exactly the recorded base, `358c163250ad3b1f62939b01ce1314a0a31a0365`, so there was no base discrepancy. On that base, an assertion against the actual imported `DeepseekV4Model._can_run_tbo` failed: non-EP `attn_tp_size=4`, `attn_dp_size=4` returned `True`, reproducing the issue-specific unsafe strategy selection.

I then checked out exact candidate commit `8574da7c83c3247d0029e5a63aeaf8dcc6521970`. The imported module resolved to `/job/repo/python/sglang/srt/models/deepseek_v4.py`. The candidate changes only Python source, a Python unit test, and reports; it changes no native source, so no native rebuild applies.

The candidate's focused regression plus existing DeepSeek-V4 RoPE tests passed (5 tests and 2 subtests). An independent 11-case truth table also passed. It covered non-EP TP2, TP4, and TP8/DP2 rejection; retention of non-EP TP1 and EP TP4; and the existing PP2, prefill-CP, decode, disabled-TBO, ineligible-batch, and missing-child gates.

The implementation matches the reported contract. `_forward_layers_tbo` builds its child length and ID metadata through the full TP group for the non-EP path, so rejecting only `backend=none && attn_tp_size>1` routes that unsupported topology through the existing synchronous forward path. The new condition does not claim to repair the collective itself.

`origin/main` had advanced to `a207786205bff0919eb2c8c9126c67f302ccff34` during review, but its intervening history contained no change to `python/sglang/srt/models/deepseek_v4.py`; the prepared base still represented the relevant failing implementation.

## Limitations

The original end-to-end reproduction requires two hosts, 16 GPUs, and DeepSeek-V4-Pro-NVFP4 weights. This environment exposed one AMD Instinct MI350X with gfx950 ISA and no required model weights. Therefore, I did not perform or claim the TP16/DP4 serving A/B, numerical model-output validation, or multi-node collective execution. The available GPU was inspected for architecture only and was not used as evidence of the fix. The review establishes the failing-before/passing-after control-flow contract against the actual implementation.
