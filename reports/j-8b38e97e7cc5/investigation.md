# DFLASH with PD disaggregation investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/36140

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1243

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The checked-out implementation still allowed DFLASH with both PD roles. Its
`SpeculativeAlgorithm.build_disagg_draft_input()` only constructs inputs for
EAGLE-family algorithms and DSPARK, returning `None` for DFLASH. DFLASH also
requires target auxiliary hidden state during prefill and maintains a separate
draft KV pool; the PD argument path had no compatibility rejection for that
state-transfer gap.

Adding only a DFLASH `spec_info` input would avoid the reported initial
`NoneType` crash while leaving the unsupported draft state absent. The narrow
correction therefore rejects DFLASH whenever `disaggregation_mode` is not
`null`, before scheduler/worker startup, with an error explaining the missing
draft KV and auxiliary-hidden-state transfer.

## Evidence

- `test_dflash_pd_gate_before.log`: the regression fails on the recorded base
  behavior because prefill/decode DFLASH configurations are accepted.
- `test_dflash_pd_gate_after.log`: the same regression passes after the guard,
  including standalone DFLASH and EAGLE/DSPARK PD boundary cases.
- `test_existing_pd_args.log`: all 13 existing PD argument-handler tests pass.
- `test_server_args_suite.log`: the broader file has 224 passes and two
  unrelated existing failures caused by the checkout's HIP prefill-CP
  deprecation behavior conflicting with older expectations.
- `gpu_environment.log`: the assigned runtime exposes one AMD Instinct MI350X.
  No GPU kernel is involved in the fail-fast argument validation.

## Limitations

The reported 8-GPU RTX 6000D, Kimi-K3/Kimi-K3-DFlash weights, multi-node PD
topology, and MoonCake RDMA setup were unavailable. Consequently this work does
not claim full-model, semantic-accuracy, NVIDIA, RDMA, or multi-node execution.
It implements and tests the issue's documented safe alternative: an actionable
startup rejection until the disaggregation protocol can transfer or rebuild
DFLASH draft state.
