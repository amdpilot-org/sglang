# Independent review of PR 1096

Reviewed candidate commit `5aad45facb10eadf8252176958378f7dd1c1576c` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

The base runner allocates only `hidden_states` for mHC pipeline proxy tensors, using `hc_hidden_size` for the folded representation. On the recorded base, GLM5 unconditionally indexed `pp_proxy_tensors["residual"]` on every non-first stage. Running the candidate's focused test against the unchanged base reproduced the reported `KeyError: 'residual'` at `glm5_next.py:984` (one failed, two passed).

The candidate makes GLM5 consume and emit a separate residual only when `config.mhc` is false. This matches the runner allocation contract and the existing folded mHC representation. At the exact candidate commit, all three focused tests and all 41 CUDA-graph buffer-registry tests passed. The imported model source was `/job/repo/python/sglang/srt/models/glm5_next.py`, confirming that tests exercised the checkout rather than an installed copy.

An independent GPU adversarial check on the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) verified that an mHC stage accepts `hidden_states` with no residual, ignores an extraneous incoming residual and emits only `hidden_states`; a non-mHC stage preserves both tensors and still rejects a missing residual. The candidate's own GPU fixture also passed with expected key sets and checksums.

Recommendation: accept. This is a source fix for the original proxy-contract mismatch, not merely test hardening. There are no source/native C++ changes, so no native rebuild was required. Full GLM-5.3-Flash-NVFP4 warmup was not possible because weights were unavailable and only one GPU was assigned. The original environment used three NVIDIA SM120 GPUs, while review execution used one AMD gfx950 GPU. Thus full serving/backend integration remains an explicit environment limitation even though the architecture-independent failure and correction were directly exercised.

Candidate evidence was copied outside the checkout to `/job/review-evidence/candidate/` before revision switching.
