# Unified-cache MAMBA host-tier investigation

Source issue: https://github.com/sgl-project/sglang/issues/33713

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3441

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Result

The reported leaf-pruning behavior did not reproduce on the prepared `main`
checkout. The current unified Python tree core treats a node as backed up when
Full KV has committed host state, demotes a backed-up Full leaf instead of
deleting it, retains it in `evictable_host_leaves`, and independently moves a
committed MAMBA state into the MAMBA host LRU. A later prefix match discovers
the host leaf and the load-back path restores both components.

This PR adds a focused regression around that complete contract. It uses the
real fixture pools and asynchronous cache controller transfers on the assigned
GPU, seeds Full KV and MAMBA temporal/convolution state with distinct numerical
markers, performs D-to-H backup, device eviction, matching, and H-to-D
load-back, and compares the restored tensors with snapshots taken before
eviction. It also checks host-leaf membership, MAMBA device/host LRU movement,
host locking, and component accounting.

Two negative cases document the intended anchor rules:

- Full host state without MAMBA host state remains a discoverable Full host
  leaf, but `build_load_back_spec` does not fabricate a MAMBA transfer.
- MAMBA host state without Full host state is incomplete tree state. Device
  eviction prunes the node, and a subsequent match reports no device or host
  hit.

## Evidence

- `raw/baseline_full_mamba_contract.txt`: four pre-existing Full+MAMBA HiCache
  tests passed on the assigned GPU, including real backup/load-back tensor
  restoration.
- `raw/regression_full_mamba_contract.txt`: all three new focused contract and
  negative-case tests passed.
- `raw/gpu_numerical_reference.txt`: PyTorch ROCm 7.2 identified the assigned
  AMD Instinct MI350X, and a GPU matrix multiplication exactly matched an
  independent CPU result.

## Limitations

The reported TP4 Ling-3.0-flash/hybrid-KDA serving reproduction was not run:
the assigned environment exposed one GPU and did not provide the model weights.
Consequently this is `candidate_verified`, not a claim that the complete
multi-GPU deployment is fixed. The deterministic tiny Llama serving fixture is
not a hybrid MAMBA/KDA architecture, so running it would only validate generic
transport/engine startup and would not add relevant evidence for this issue.
No native C++ or FlyDSL code changed, so no native rebuild was applicable.

