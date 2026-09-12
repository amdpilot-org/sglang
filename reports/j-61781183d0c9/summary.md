# Correction generation 1

Candidate: https://github.com/amdpilot-org/sglang/pull/1145 at `803c48fc6fa1441b4e79ccd83de0bb207361947e`

Independent review: https://github.com/amdpilot-org/sglang/pull/1242

The review counterexample was reproduced through `FusedMoE._weight_loader_impl`.
For a GPTQ GROUP `w2_scales` parameter with `load_full_w2=True`, TP size 2,
and rank 1, the candidate copied only the checkpoint's second half into the
start of the full destination and left its remaining rows untouched.

The correction forwards the existing `load_full_w2` parameter attribute from
the generic GROUP/BLOCK scale branch to the candidate's full-table loader.
The regression uses the production dispatch and independently checks both the
act-order full-table case and the non-act-order sharded case.

The original NVIDIA CUDA Marlin TP=2 serving workload and model weights were
not available. No end-to-end serving, kernel, or NaN-logit reproduction is
claimed.
