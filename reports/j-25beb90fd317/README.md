# Investigation: sparse destination-aware MoE dispatch

Upstream issue: https://github.com/sgl-project/sglang/issues/36820

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2684

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Outcome

The reported `a2a=none` behavior is reproduced, but the requested feature is
not implemented in this contribution. The prepared node exposes one AMD
Instinct MI350X. That is enough to execute the checked-out Qwen3 method and a
local GPU numerical identity check, but it cannot qualify a four-rank sparse
transport protocol, prove that an inactive rank performs no network payload
transfer, exercise cross-node failure ordering, or compare against the stated
NVIDIA L4/no-RDMA deployment.

Implementing only a destination planner, metrics, or a host-coordinated toy
protocol would leave the central contract unimplemented and could introduce a
deadlock in request-, layer-, microbatch-, or CUDA-graph-dependent routing. No
such partial runtime path was added.

## Reproduction

`Qwen3MoeSparseMoeBlock.forward_normal` computes a local expert result and then,
whenever `ep_size > 1`, invokes `moe_expert_parallel_all_reduce` unless a
separate downstream fusion/reduce-scatter path replaces that reduction. It does
not inspect the selected experts or the local contribution before entering the
collective.

`test_qwen3_none_path_contract.py` calls the actual method from the prepared
checkout. The adversarial zero-local-contribution case and a nonzero case both
record exactly one EP reduction. EP size one does not reduce. A fourth test
shows that the existing downstream-fusion skip is not destination-aware
dispatch.

Command and raw output:

```text
PYTHONPATH=/job/repo/python \
XDG_CACHE_HOME=/tmp/amdpilot-repo-j-25beb90fd317/test-cache \
TRITON_CACHE_DIR=/tmp/amdpilot-repo-j-25beb90fd317/triton-cache \
/tmp/amdpilot-repo-j-25beb90fd317/venv/bin/python -m pytest -q \
  reports/j-25beb90fd317/test_qwen3_none_path_contract.py
```

See `pytest-contract.log`: 4 passed.

## Existing related work inspected

- Upstream PR 36861 fixes logical-to-physical expert-location routing in Qwen3
  normal mode and explicitly excludes sparse communication. It is open and is
  not present in this prepared base.
- Upstream PR 32329 proposes an optional NCCL EP decode backend. It is open,
  NVIDIA/NCCL-specific, and lists prefill, CUDA graphs, and deterministic
  fallback as follow-ups. It does not establish the requested L4/no-RDMA
  support.
- Upstream PR 32963 changes/fuses post-expert reductions but retains a group
  reduction; it is not inactive-rank skipping.
- Existing DeepEP, Mooncake, Mori, NIXL, PPLX, FlashInfer, and MegaMoE paths are
  opt-in dispatcher transports with distinct platform/topology constraints.
  Their presence does not change the standard `a2a=none` behavior reproduced
  here.

## GPU evidence and its boundary

`gpu_sparse_combine_reference.py` ran on the assigned MI350X. It constructs
four per-rank partial outputs, makes two inactive ranks exactly zero, and shows
that summing only active contributions is numerically identical to summing all
partials. The GPU result is checked against a NumPy/CPU float64 sum. The ideal
payload element count falls from 896 to 448 for this synthetic active set.

See `gpu-reference.log`. This validates only the reduction algebra. It does not
execute distributed communication, and is not evidence for payload bytes,
latency, deadlock freedom, CUDA graph replay, server token/logprob equivalence,
or L4 compatibility.

## Unimplemented and unverified contract

- logical expert to destination-rank mapping integrated with current and
  remapped expert placement;
- compact activation dispatch and selected contribution return;
- a deadlock-free control protocol and sequence/layer/microbatch identity;
- inactive-rank removal from payload transfer and the critical-path reduction;
- dense-fanout threshold and safe collective fallback;
- per-layer active-fanout, byte, and sparse-hit metrics;
- CUDA graph capture/replay and overlap modes;
- hybrid EP/TP/DP-attention group semantics, empty ranks, uneven tokens, expert
  replication, speculative decoding, prefill, and multi-request concurrency;
- deterministic server token-ID and server-authored-logprob equivalence;
- network counter and latency measurements on four cross-node NVIDIA L4 ranks.

The tiny Llama fixture mentioned in the task was not used: it cannot qualify
Qwen3 MoE architecture semantics or a distributed EP workload, and this node
cannot launch the required multi-rank topology.

No native sources changed, so no native rebuild was applicable.
