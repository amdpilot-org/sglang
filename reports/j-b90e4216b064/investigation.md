# Investigation: multimodal pickle amplification

Upstream issue: https://github.com/sgl-project/sglang/issues/33388

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1897

At base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the exact 512-image
fixture reproduced 518.307x amplification. Each 2,048-byte feature slice
retained its 1,048,576-byte parent storage, and pickle emitted that storage for
each split item.

The fix materializes only CPU tensor slices in the common slicing helper and
routes the simple-split tensor branches through it. This covers features,
precomputed embeddings, item-aligned metadata, and feature/patch-aligned
metadata. Accelerator tensors remain views to preserve IPC/shared-memory
behavior.

Raw before/after reproduction, focused test output, and GPU boundary evidence
are retained alongside this report. The exact fixture falls from 543,484,451
pickle bytes (518.307x) to 1,375,278 bytes (1.312x).

This was not a full model or HTTP serving reproduction. It directly exercises
the reported implementation and serialization mechanism without requiring
weights. The accelerator boundary check ran on the assigned AMD Instinct
MI355X; it verifies values and storage sharing, not a CUDA H200 deployment.
