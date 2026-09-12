# EAGLE greedy TP synchronization investigation

The prepared base already includes the issue's ROCm fix. Upstream PR
https://github.com/sgl-project/sglang/pull/34238 added rank-0 synchronization
of the finalized EAGLE verify decision, and upstream PR
https://github.com/sgl-project/sglang/pull/35195 scoped the greedy-path addition
to HIP after an XPU incompatibility was reported. At base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`, `eagle_sample` broadcasts
`predict`, `accept_index`, and `num_correct_drafts` whenever `_is_hip` and the
selected TP group's world size is greater than one.

The added regression calls the actual `eagle_sample` control flow with GPU
logits for two logical ranks. The second position is near-tied in opposite
directions, producing local argmax decisions `[1, 2]` and `[1, 1]`. A replay
group models the semantics of rank-0 broadcast without claiming a real TP=2
run. On the prepared source, the three finalized decision tensors agree. When
only the existing HIP broadcast block is temporarily removed, the main test
and DP-attention routing test fail; the TP=1 boundary continues to pass.

The assigned hardware was one AMD Instinct MI355X (`gfx950`) GPU. This was
enough to execute the tensor/argmax path and compare it with NumPy CPU
references, but not to launch a real two-rank collective, reproduce the
subsequent deadlock, or validate a full EAGLE model. No model weights were
used. No native source changed or required rebuilding.

Raw test and GPU evidence are stored beside this report.
