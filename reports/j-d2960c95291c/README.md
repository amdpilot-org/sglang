# RVV feature-port validation record

This contribution ports the full scope described by the source issue and its
three linked open changes onto base `358c163250ad3b1f62939b01ce1314a0a31a0365`:

- Phase 1: RVV decode/extend attention, GEMM and INT8 GEMM, norm, activation,
  RoPE, operator registration, and native numerical tests.
- Phase 2: platform/capability detection, weight packing, Python operator
  dispatch, attention backend registration, and integration tests.
- Phase 3: the RISC-V package definition, Dockerfile, and platform guide.

The current SGLang tree moved server-argument resolution out of
`server_args.py`. The port therefore registers `rvv` in
`arg_groups/choices.py` and selects it through `arg_groups/platform_hook.py`.
Regression tests cover RVV auto-selection and verify that an explicit CPU
backend remains authoritative.

## Hardware and compiler boundary

The prepared node reports `x86_64` on an AMD EPYC 9965. It is not a RISC-V
host and exposes no RVV 1.0 execution path. The installed ROCm Clang 22 accepts
a RISC-V target triple but omits the required `riscv_vector.h`; the retained
`rvv-vector-math-clang.log` shows the compiler failure. Consequently no native
RVV library, instruction disassembly, independent kernel numerical comparison,
Qwen GSM8K run, or Banana Pi performance comparison was produced.

The passing x86 tests qualify compatibility, dispatch guards, fallback behavior,
and test collection only. They do not qualify native RVV correctness or the
end-to-end feature on its target architecture.
