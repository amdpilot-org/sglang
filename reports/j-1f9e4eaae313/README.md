# Investigation: FlashInfer JIT cannot find `-lcuda`

Upstream issue: https://github.com/sgl-project/sglang/issues/36058

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1261

## Finding

No SGLang source defect was established on the prepared checkout. The report is
an environment-dependent CUDA host-link failure inside FlashInfer. The exact
reported dependency (`flashinfer-python==0.6.17`) and the version pinned by this
checkout (`0.6.18`) both generate these default link flags:

```text
-L$cuda_home/lib64 -L$cuda_home/lib64/stubs -lcudart -lcuda
```

Thus FlashInfer already searches the standard CUDA toolkit stub directory. The
reported error means the selected CUDA installation did not provide a linkable
unversioned `libcuda.so` in either default directory, and the host compiler did
not find one in its own search path. A runtime driver file named only
`libcuda.so.1` does not satisfy the linker's `-lcuda` lookup.

The deterministic matrix in `reproduce_linker_matrix.sh` uses FlashInfer's
actual search-path ordering. It shows:

1. no `libcuda.so`: the reported linker error occurs;
2. only `libcuda.so.1`: the same error occurs;
3. `$CUDA_HOME/lib64/stubs/libcuda.so`: linking succeeds;
4. an unversioned system driver library plus an explicit `-L` directory:
   linking succeeds.

This supports fixing the affected environment by installing/restoring the CUDA
toolkit's driver stub or exposing the directory containing an unversioned
driver linker library through FlashInfer's documented
`FLASHINFER_EXTRA_LDFLAGS` hook. Adding another SGLang linker flag without the
reporter's actual library location would merely guess a host-specific path.

## Scope and limitation

The assigned device is one AMD Instinct MI355X (`gfx950`) with ROCm 7.2. The
report requires NVIDIA T4 (`sm_75`), CUDA 12.8/12.9, FlashInfer JIT, and the
reporter's unspecified model weights. This environment has no CUDA toolkit,
NVIDIA device, FlashInfer installation, or those weights, so the original
server launch and GPU kernel execution cannot be reproduced honestly here.
The linker fixture validates only the host linker's handling of the exact
library names and search-path forms; it does not validate model serving,
FlashInfer kernel compilation, semantic accuracy, or a T4 execution path.

## Evidence

- `flashinfer-0.6.17-cpp_ext.txt`: relevant source from the exact reported wheel.
- `flashinfer-0.6.18-cpp_ext.txt`: relevant source from the checkout's pin.
- `linker-matrix.log`: deterministic missing/stub/system-library boundaries.
- `gpu-inventory.log`: assigned hardware and framework inventory.
- `issue.json`: source issue state/body captured during investigation.
