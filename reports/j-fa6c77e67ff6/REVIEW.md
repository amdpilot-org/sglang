# Cascade-attention correction generation 1

Candidate: https://github.com/amdpilot-org/sglang/pull/3146 at `8f1eaa1011c775914123ce8ee3021e2eb2b82982`

Independent review: https://github.com/amdpilot-org/sglang/pull/3192

The review's counterexamples reproduce. PR 3146 implements a coherent eager-only
shared-prefix path, but it deliberately leaves CUDA-graph replay and several attention
configurations on ordinary decode. Its float64 GPU test checks the split-softmax identity,
not FlashInfer integration. The prepared MI355X/ROCm environment has no FlashInfer module,
so actual NVIDIA wrapper/kernel execution and serving performance cannot be established.

This correction retains the eager implementation, removes the old raw-log whitespace
failure, and adds an NVIDIA-CUDA-only integration regression. That test constructs the
same shared and per-request page tables as the backend, calls both real
`BatchPrefillWithPagedKVCacheWrapper.forward_return_lse` paths, invokes the production
merge function, and compares with an independent PyTorch attention reference. It skips
honestly on the prepared node rather than turning unavailable hardware into a source
change.

Raw failing-before, passing-after, and environment outputs are retained in `raw/`.
