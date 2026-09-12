# Independent review evidence

Reviewed PR: https://github.com/amdpilot-org/sglang/pull/3117  
Exact commit: `5ce4f2e1bdd7f4f94168b4d7c56213775d57474f`  
Upstream issue: https://github.com/sgl-project/sglang/issues/32200  
Mirror issue: https://github.com/amdpilot-org/sglang/issues/3131

## Finding

Recommendation: **unverified**. The candidate fixes the concrete allocator
scoping defect reported by the prior independent review, and no additional
source-level defect reproduced in the available environment. It cannot be
classified as a fully verified original-issue fix because the required NCCL
2.30+/NVLink two-rank RMA handshake was not executable.

## Failing before

At recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the extracted
candidate regression fails during collection because
`NCCL_WIN_COLL_SYMMETRIC` is absent. A direct contract probe additionally
records:

```text
rma_table None
bindings {'ncclPutSignal': False, 'ncclSignal': False,
          'ncclWaitSignal': False, 'ncclWinGetUserPtr': False,
          'ncclGetPeerDevicePointer': False, 'ncclMemAlloc': False,
          'ncclMemFree': False, 'ncclCommInitRankConfig': True}
allocator_get_windows False
rma_communicator_error ModuleNotFoundError
```

This reproduces the original issue rather than an unrelated startup smoke.

## Candidate validation

The exact candidate suite reports `19 passed, 2 skipped`. The skipped tests
are the real two-GPU RMA tests.

A fresh build under
`/tmp/amdpilot-repo-j-9b5b0f3eccdb/review-tmp/symm_allocator` compiled the
candidate allocator source and linked it with `-lrccl`. The rebuilt library
exports both accessors and the corrected loader publishes both ctypes objects:

```text
native_get_windows_symbol= True
native_clear_windows_symbol= True
module_get_windows_func= <_FuncPtr ...>
module_clear_windows_func= <_FuncPtr ...>
```

The empty `collected_windows=[]` for the probe's fake communicator is expected:
no allocation was registered for that communicator. The candidate's mocked
loader regression independently supplies a synthetic non-NULL native handle
and verifies it is returned as `[0xA11CE]`.

Three independent adversarial tests passed: a synthetic non-NULL all-rank
handle set reaches the handshake and sets `rma_available=True`; a NULL handle
skips the handshake and frees memory; and a verification exception falls back
with window deregistration and memory cleanup.

The NCCL 2.30.7-1 source header and implementation were retained outside the
checkout and used to compare the candidate's ctypes layouts and call ordering.

## Environment boundary

The host has one AMD Instinct MI350X (`gfx950`), torch
`2.11.0+rocm7.2`, and RCCL `2.27.7`. Its RCCL exports the allocator and
configured-init symbols but not the NCCL 2.30 RMA primitives. Consequently no
candidate GPU RMA operation ran, and this review does not claim proxy data
movement, signal ordering, or received-data correctness on NVIDIA/NVLink.

Raw logs, extracted tests, upstream source snapshots, symbol listings, and the
independent adversarial test are retained at
`/job/review-evidence-j-9b5b0f3eccdb` across revision switches.
