# Independent review of PR 2657

Candidate: `b8b17e6d25941987360e9dad6d5d196c66b25cad`

Recorded comparison base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate is a source-level full fix for the
reported default-selection contract. It is not merely test hardening: the base
renders DSV4 in chat mode with no explicit effort, while the candidate defaults
only the DSV4 encoder to thinking/high and scopes the ROCm workaround override
to the reported `gfx950-rocm720` image stage.

## Independent findings

- On the recorded base, an official-profile DSV4 request with both semantic
  environment variables unset produced `thinking_mode=chat`,
  `reasoning_effort=None`, no `<think>` tag, and no high-effort instruction.
- At the exact candidate commit, its focused suite passed: 138 tests and 70
  subtests. An independent matrix also passed for official/preview profiles,
  explicit false, explicit empty effort, request-over-environment precedence,
  and non-DSV4 behavior.
- `SGLANG_DEFAULT_THINKING` controls chat rendering mode. The candidate keeps
  its new unset default local to `chat_encoding_spec == "dsv4"`; an explicitly
  configured global value still applies to other encoders.
- `SGLANG_DSV4_REASONING_EFFORT` is consulted only in the DSV4 path after a
  request-level value. Unset now selects `high`; an explicitly empty value is
  preserved and falls through to the checkpoint profile's encoder default.
- `SGLANG_USE_ROCM700A` is read at import time by `dp_attention.py` and changes
  the CUDA-graph DP padding default. On this MI355X/gfx950 host, isolated
  processes measured `0 -> MAX_LEN` and `1 -> SUM_LEN`. It is a ROCm 7.0.0-alpha
  workaround selector, not a GPU ISA identifier.
- The candidate changes no native/C++/FlyDSL source. No native rebuild was
  applicable. Python imports resolved to `/job/repo/python/sglang/...`; AITER
  resolved to the prepared private-runtime cache.

## Scope and limitations

The assigned hardware is an AMD Instinct MI355X reporting
`gfx950:sramecc+:xnack-`, with Torch 2.11.0+rocm7.2 and HIP 7.2.26015. A real GPU
calculation matched an independent CPU result with maximum absolute error
`4.76837158203125e-07`.

DeepSeek-V4-Pro weights were unavailable, so semantic quality, accuracy,
throughput, and distributed model execution were not validated. The production
Docker image was not built; stage inheritance was checked from the exact
Dockerfile and by the candidate regression. No distributed DP-attention job was
run, so the flag evidence establishes branch selection rather than distributed
correctness or performance. These limitations do not produce a source-level
counterexample to the requested default/override contract.
