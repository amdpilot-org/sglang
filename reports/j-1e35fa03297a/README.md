# Investigation of DeepEP low-latency initialization during graph capture

Upstream issue: https://github.com/sgl-project/sglang/issues/29942

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2409

## Result

No SGLang defect was reproduced and no runtime source correction is justified
by the available evidence. The prepared source already executes two complete
model forwards before entering the CUDA graph context in
`FullCudaGraphBackend.capture_one`. Any process-wide `DeepEPBuffer` created by
the first MoE dispatch is therefore created during an eager warmup, not by the
subsequent captured forward.

This ordering is not a post-report fix. Upstream commit
`f8f2870a84a4` (2026-08-06, before the issue's 2026-08-12 update) has the same
two eager `forward_fn()` calls before `graph_ctx(...)`. The prepared base keeps
that ordering and adds graph-pool measurement around it.

The issue's latest independent evidence also shows that
`deep_ep.cpp:230 'invalid argument'` is not diagnostic of lazy initialization:
an NVSHMEM physical-allocation OOM returned a null pointer, followed by the
same error from `cudaMemset(nullptr, ...)`. For that reported DSpark boundary,
the low-latency RDMA hint was 21.33 GiB per rank while only about 19.2 GiB was
free. That is not proof of the Kimi reporter's root cause, but it prevents the
shared line-number signature from justifying an eager-init patch by itself.

## Focused validation

`verify_capture_order.py` invokes the actual prepared
`FullCudaGraphBackend.capture_one` on the assigned gfx950. Its forward lazily
allocates a GPU workspace, records whether each invocation is capturing, and
computes an exact reference result after graph replay. It measured:

```text
device=AMD Instinct MI350X
torch=2.11.0+rocm7.2 hip=7.2.26015
capture_state=[False, False, True]
lazy_allocations=1
max_abs_error=0.0
```

Thus the allocation occurred once during the first eager invocation; the two
warmups were outside capture and only the third invocation was captured.

Raw runtime output is retained at
`/tmp/amdpilot-repo-j-1e35fa03297a/evidence/gfx950_actual_backend_capture_order.log`.

## Limitations

The assigned host has one AMD gfx950 GPU. The prepared environment does not
contain the `deep_ep` Python extension. The Kimi K2.6 W4A8 weights, CUDA/H800,
NVSHMEM/InfiniBand setup, two nodes, and the required 16-GPU PP=2/TP=8/DP=8/EP=8
topology are unavailable. Consequently this investigation does not claim a
full model, DeepEP collective, multi-rank, or multi-node reproduction, and it
cannot distinguish an allocation OOM from another CUDA/NVSHMEM failure in the
original reporter's environment.

The gfx950 test validates only the checked-in SGLang graph-boundary ordering
and GPU graph execution. Reproducing the original workload remains necessary
to attribute its failure conclusively.

