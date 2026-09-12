# Investigation result

The prepared base already contains the issue-specific implementation fix. Upstream PR
[#38830](https://github.com/sgl-project/sglang/pull/38830), merged before base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`, changed
`expert_pack_mxfp4` from `torch.utils.cpp_extension.load()` to SGLang's
`load_jit()`.

The distinction is material to the reported hang:

- The prepared PyTorch 2.11 `FileBaton` treats existence of a lock file as
  ownership. A deterministic reproduction with an ownerless file remained in
  `FileBaton.wait()` until an external two-second timeout (`exit 124`).
- Current `load_jit()` opens a persistent `.lock` path and obtains
  `fcntl.flock(LOCK_EX)`. The kernel releases ownership when the descriptor or
  process dies, regardless of whether the path remains on disk.

This change adds regression coverage for the two missing boundaries alongside
the existing live-owner exclusion test: an already-present ownerless lock file
does not block, and killing the process holding a real advisory lock permits an
immediate later acquisition.

## Hardware limitation

The actual test was launched with a private cold JIT cache on the assigned AMD
Instinct MI355X (`gfx950`). It reached `hipcc` with
`--offload-arch=gfx950:sramecc+:xnack-`, then failed because the source includes
the NVIDIA header `cuda_bf16.h`. No MXFP4 kernel executed, so this report does
not claim GPU numerical validation or an H200 reproduction. The raw compiler
output is retained in `raw/mxfp4_gfx950_cold.txt`.

## Evidence

- `raw/expert_pack_before_38830.py`: source immediately before the merged port,
  showing `torch.utils.cpp_extension.load`.
- `raw/implementation_transition.txt`: old/new loader references and current
  lock implementation location.
- `raw/legacy_filebaton_reproduction.txt`: deterministic stale-file timeout.
- `raw/jit_cache_full.txt`: 44 passing JIT cache tests after the regression.
- `raw/mxfp4_gfx950_cold.txt`: cold-cache gfx950 compilation attempt and exact
  architecture blocker.
