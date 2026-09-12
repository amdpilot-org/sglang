# Investigation result

Upstream issue: https://github.com/sgl-project/sglang/issues/31347

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2757

The prepared main checkout already contains the relevant solution. Upstream PR
https://github.com/sgl-project/sglang/pull/34274 merged on 2026-08-14 as
`b7847268632ae8d684ecb0e2395aa3db765ab892` and replaced tvm-ffi's in-place
build with SGLang's content-addressed JIT cache.

In the checked-out implementation, `load_jit` creates
`.staging-<uuid>` under the build-key scope, compiles and loads the library
there, and only then calls `commit_build`. Publication uses `os.rename` to move
the complete staging directory to an immutable dependency-key leaf. If another
host publishes the same leaf first, the loser keeps the winner's leaf and
discards its private staging directory. Thus a non-coordinating NFS `flock`
can at worst permit duplicate work; it no longer permits two nodes to rewrite
the same `cuda_0.o` or final `.so` while a linker or loader reads it.

The existing focused regression `test_publish_loses_the_race_gracefully`
exercises the important boundary directly: a winner leaf already exists, a
loser has a distinct staged library, and publication preserves the winner.
The independent lock test verifies same-host exclusion, while the implementation
comments explicitly make atomic publication—not the lock—the correctness
mechanism. All 42 focused cache tests passed.

No duplicate source fix was added. The older open PR
https://github.com/sgl-project/sglang/pull/31725 targets the former tvm-ffi
loader and is superseded in current main by the broader merged cache rewrite.

## Hardware limitation

The assigned device was one AMD Instinct MI350X (`gfx950`). A cold JIT attempt
confirmed that the Hadamard build uses a UUID-private staging directory, but
the CUDA-oriented source failed before kernel launch because
`fast_hadamard_transform_common.h` includes `cuda_bf16.h`, unavailable in the
prepared ROCm environment. Consequently there is no GPU numerical claim. The
required two-node shared-NFS/CUDA model environment was also unavailable, so
the original ESTALE report was not reproduced end to end.

Raw command output is retained under `raw/`, including the focused CPU suite,
GPU inventory, related-PR metadata, and the complete first compiler failure.
