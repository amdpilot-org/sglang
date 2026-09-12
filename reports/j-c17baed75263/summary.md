# Independent review of PR 1319

Reviewed `https://github.com/amdpilot-org/sglang/pull/1319` at exact commit `539ca7f4667320c270390c9f821715c2297b71ae` against upstream issue `https://github.com/sgl-project/sglang/issues/36822` and mirror issue `https://github.com/amdpilot-org/sglang/issues/1362`.

Recommendation: **accept**. The candidate is a source-level full fix for the original loader/allocation contract. On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the candidate regression failed 5 of 7 cases: BF16 scales remained FP16, the `desc_act` TP shape decision was reversed, `_load_w2` lacked full-table support, and production `_weight_loader_impl` on TP rank 1 copied the second shard into the start of a full destination. At the exact candidate, all 7 regression cases passed.

Independent adversarial checks covered TP ranks 0 and 1, `desc_act` both false and true, `w2_scales` and `w2_qzeros`, BF16 dtype/shape/metadata allocation, and production `_weight_loader_impl` dispatch. They passed on CPU and on the assigned AMD Instinct MI355X (`gfx950`) GPU. Imports resolved to `/job/repo/python/sglang` and the reviewed `FusedMoE` source in `/job/repo/python/sglang/srt/layers/moe/fused_moe_triton/layer.py`.

No native files changed, so no native rebuild was applicable. The prepared host has one AMD ROCm GPU, not the issue's two NVIDIA CUDA GPUs, and the reported model weights are unavailable. Therefore the exact CUDA Marlin kernel, TP=2 distributed serving process, semantic output, and NaN-logit symptom were not executed end to end. These are environment limitations, not observed source counterexamples.

Raw command output is retained in this report directory. The review branch contains no duplicate candidate patch.
