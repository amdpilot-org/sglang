# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/33577

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1807

The recorded base still contained both ordering/state defects from the report.
`GroupCoordinator._all_to_all_single` selected the torch.distributed fallback
whenever the normally-disabled pynccl communicator was disabled, while the
neighboring pynccl collectives temporarily enable it with `change_state`.
Also, the 4 GiB symmetric-memory reservation ran from `capture_cuda_graphs`,
after `ModelRunner.alloc_memory_pool` had profiled available memory and
allocated the KV cache.

The patch temporarily enables pynccl for all-to-all and restores its prior
state on exit. It also moves the existing symmetric-memory reservation to the
start of `alloc_memory_pool`, immediately before KV-cache profiling. The
reservation helper's existing draft-worker, feature, and size guards remain
unchanged.

The regression was executed against the base implementation by reversing the
tracked source patch while retaining the new test. It failed in both defect
paths (2 failed, 2 passed), as captured in `raw/failing-before.txt`. With the
patch restored, all four cases passed (`raw/passing-after.txt`). The independent
cases cover an already-enabled communicator, the no-pynccl fallback, state
restoration, and the reservation/profile ordering.

The assigned environment has one AMD Instinct MI350X (gfx950) using ROCm 7.2,
not the report's eight-rank CUDA/NCCL 2.27.7 setup. A single-device numerical
probe ran successfully and is retained in `raw/gpu-probe.txt`, but it does not
exercise DCP all-to-all and is not claimed as reproduction of the reported
crash. Model weights were not available or needed for the focused regressions.
No native source was changed or rebuilt.
