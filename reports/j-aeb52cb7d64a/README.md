# Custom all-reduce concurrent-stream investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/31117

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2295

The recorded base still had no ordering guard for the per-communicator custom
all-reduce rendezvous state. An upstream candidate exists at
https://github.com/sgl-project/sglang/pull/31135 but was still open and absent
from main during this investigation.

This change adds the narrow host-side part of that correction: when eager
all-reduces on one communicator switch GPU streams, an event orders the new
stream behind the previous stream. Reuse of one stream remains a raw stream
pointer comparison. Graph capture is not modified because concurrent graph
replay makes no host call and therefore cannot be protected by a host guard.

The original failure requires two NVIDIA GPUs with NVLink and CUDA custom
all-reduce. The assigned environment contains one AMD Instinct MI350X (gfx950),
so the A100 two-rank deadlock was not reproduced and the CUDA kernels were not
compiled or executed. The committed unit regression verifies the ordering
contract and its same-stream/capture boundaries. A real gfx950 two-stream probe
also verified that the guard's event produces the expected numeric order.

Raw command output is retained in `raw/`.
