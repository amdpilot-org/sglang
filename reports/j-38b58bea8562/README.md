# Mooncake failed-session recovery correction

Upstream issue: https://github.com/sgl-project/sglang/issues/37022

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1220

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1101 at `0aeacd1b2bf8d0a0920c0a2beb6db2393d88311c`

Independent review: https://github.com/amdpilot-org/sglang/pull/1188

## Result

The review's two concrete control-plane counterexamples reproduced against the
candidate's unchanged production implementation. With the environment variable
unset, failed-session probing was disabled. A deterministic interleaving also
showed that a successful probe started for failure count 1 erased a newer
failure count 2 for the same session.

The correction enables failed-session probing by default while preserving
`SGLANG_ENABLE_FAILED_SESSION_PROBE=0` as an explicit opt-out. A probe pass now
snapshots each failed session's failure count and only clears the blacklist if
that count is unchanged when the probe completes. The candidate's successful,
failed, exceptional, and native-wrapper probe tests are retained alongside the
new default, opt-out, and stale-probe regression cases.

Before the source correction, the focused suite reported 2 failures, 9 passes,
and 3 passing subtests. After the correction and opt-out boundary case, it
reported 12 passes and 3 passing subtests.

## Limitations

This corrects recovery behavior after a Mooncake transfer failure; it does not
identify or correct the cause of recurring transfers failing in the reported
5-prefill/3-decode GLM5.2 NVIDIA/XCCL multi-node RoCEv2 deployment. That model,
network topology, CUDA/XCCL stack, and weights were unavailable. The prepared
host exposed one AMD Instinct MI355X (`gfx950`) GPU. No GPU workload was run as
substitute evidence for the unavailable distributed deployment, and no native
library was changed or rebuilt.

Raw command output is retained in `raw/`.
