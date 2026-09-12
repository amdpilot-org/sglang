# NCCL RMA window binding correction evidence

Candidate: https://github.com/amdpilot-org/sglang/pull/2952 at
`34d861306b4fdce616d2260d9d9f52655d6ef145`

Independent review: https://github.com/amdpilot-org/sglang/pull/3021

## Failing before

The candidate's focused suite passed (`18 passed, 2 skipped`), but a clean native
allocator rebuild followed by `reproduce_window_binding.py` exited 1:

```text
[1/2] c++ ... -c /tmp/symm_allocator/main_hip.cpp -o main_hip.o ...
[2/2] c++ main_hip.o -shared -lrccl ... -o nccl_allocator.so
native_library=/tmp/symm_allocator/nccl_allocator.so
native_get_windows_symbol= True
native_clear_windows_symbol= True
module_get_windows_func= None
module_clear_windows_func= None
collected_windows= []
AssertionError: assert allocator._get_windows_func is not None
```

This separates the deterministic Python scoping defect from native compilation
or symbol availability: the rebuilt library exported both accessors.

## Passing after

After declaring both accessor bindings global, the focused suite reports
`19 passed, 2 skipped`. The same native probe exits 0:

```text
native_library=/tmp/symm_allocator/nccl_allocator.so
native_get_windows_symbol= True
native_clear_windows_symbol= True
module_get_windows_func= <_FuncPtr object at ...>
module_clear_windows_func= <_FuncPtr object at ...>
collected_windows= []
```

The empty list is expected for the arbitrary, unregistered communicator used by
the probe. The regression test additionally installs a native-compatible test
accessor returning `0xA11CE` and verifies `_collect_windows_for_comm()` surfaces
that exact non-null handle.

## Hardware limitation

The assigned machine exposes one AMD Instinct MI350X (`gfx950`) through ROCm
7.2. It cannot execute or qualify the required two-rank NVIDIA NCCL 2.30/NVLink
ring `put_signal`/`wait_signal` handshake. The two hardware tests remain skipped;
this is not presented as evidence that the NVIDIA RMA success path works.
