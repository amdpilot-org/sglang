# Independent review of PR 2448

Upstream issue: https://github.com/sgl-project/sglang/issues/30322

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2379

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2482

Candidate: https://github.com/amdpilot-org/sglang/pull/2448 at
`a171524ccc35794fcd1bbd5501cd8ea61d0a3bff`.

## Recommendation

**Accept as test-only hardening, not as proof that the original distributed
issue is fully resolved.**

The candidate changes no production source. Its only executable change adds
three CPU mock tests for the current decode HiCache restore state machine. All
five tests in the modified module, including the three candidate tests, also
pass on the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` after
temporarily applying only the candidate test file. Therefore these are useful
coverage tests, but not a failing-before/passing-after regression and not a
candidate implementation fix.

The candidate's prose is appropriately limited: it reports `not_reproduced`
and says the prepared source already contains restore gating. Source inspection
confirms that `HiCacheRestoreGatedKVReceiver` hides network success while local
restore is pending, `_try_hicache_queue_load_back` waits for L3 prefetch and
checks full promised-prefix coverage, and `pop_transferred` aborts a request
whose local restore becomes `FAILED`.

At the exact candidate commit, its claimed focused suite passed 25 tests.
Independent cases also passed for a 63/64-token incomplete restore becoming
`FAILED`, mixed L1=8/L2=16/L3=40 coverage producing exactly 56 restored
indices, and an underlying transport failure remaining visible through the
metadata gate.

## Scope and limitations

The original report requires eight NVIDIA H20 prefill ranks with CP=8, eight
decode ranks with DP=8, Mooncake transfer, persistent HiCache data, and a decode
process restart. This environment exposes one AMD Instinct MI355X (`gfx950`)
through Torch 2.11.0+rocm7.2 and provides neither that distributed topology nor
the model/storage fixture. The tests exercise CPU control flow and index
accounting only. They do not validate Mooncake transport, persistent-storage
recovery, restart races, H20/CUDA behavior, full-model semantics, or a
multi-node workload.

The imported `sglang` and `decode_hicache_mixin` modules both resolved to
`/job/repo/python`, so candidate tests used checkout source rather than an
unrelated installed package. The candidate changes no C++/HIP/CUDA/native
source; no native rebuild was applicable. GPU availability and architecture
were recorded, but no issue-specific GPU kernel was executed.

Raw commands and output are retained under `raw/`. In particular,
`base-with-candidate-regression.txt` demonstrates that the candidate tests
already pass on the base, and `candidate-adversarial.txt` contains the
independent boundary result.
