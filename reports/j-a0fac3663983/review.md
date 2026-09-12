# Independent review of amdpilot-org/sglang PR 1001

Reviewed exact candidate commit `2af74468719b913fa5b64e0096c035f2e3d40ea7` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept**. The candidate fully resolves the reported latent Spark2.5 attention-gate activation mismatch.

The base implementation's real `Spark2_5Attention.forward` unconditionally called `torch.sigmoid(g.float())`. A config constructed with `gate_attn_act_mode="silu"` retained that value through `PretrainedConfig`, but the decoder never passed it to attention and attention had no branch for it. The submitted regression failed on the base at collection because the fixed activation selector was absent; independent source and numerical evidence confirmed the hardcoded path and a maximum absolute sigmoid-versus-SiLU difference of 19.0 on the adversarial input range used in review.

At the candidate revision, the config declares and stores the field, the decoder passes it to `Spark2_5Attention`, and the real forward path selects sigmoid or SiLU after conversion to float32 and rejects unsupported values. The candidate's four tests passed. Independent tests additionally covered float16, bfloat16, and float32 inputs; empty, mixed-case, unknown, and null modes; constructor propagation; and the actual attention forward gating block. CPU and assigned-GPU results exactly matched independent PyTorch references.

GPU evidence came from one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) under PyTorch `2.11.0+rocm7.2`. No native files changed, so no native rebuild was applicable. No Spark-X2.5 weights were available, and therefore this review does not claim a full-model generation or semantic-accuracy comparison. The tiny Llama fixture was intentionally not substituted because it cannot qualify the Spark2.5 architecture path. No multi-GPU or multi-node validation was performed.

Raw command output was preserved outside the revision-switching checkout at `/job/review-evidence-j-a0fac3663983/`.

Original issue: https://github.com/sgl-project/sglang/issues/37639

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1033
