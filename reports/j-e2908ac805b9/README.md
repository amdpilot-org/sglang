# Correction review: Kimi-K3 TP embedding collective divergence

Upstream issue: https://github.com/sgl-project/sglang/issues/37393

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1144

Candidate: https://github.com/amdpilot-org/sglang/pull/1035 at
`5247b13d80bb3e7549c5517a7da17ca469ac8cd0`

Independent review: https://github.com/amdpilot-org/sglang/pull/1115

## Outcome

The candidate's useful unit coverage is retained, but its `candidate_verified`
classification is rejected. The candidate changes no runtime source and its
exact five tests pass unchanged on the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The prepared base already contains
the DSpark speculative-decision broadcasts that were absent from the reported
v0.5.18 source. Those broadcasts are a plausible mitigation for one divergence
path, not evidence that they caused either recorded production sequence or
that all TP state-divergence paths are closed.

No additional runtime correction is justified by the available evidence.
Changing `VocabParallelEmbedding` cannot make unequal collective shapes valid,
and adding an unconditional preflight collective would add serving overhead
without identifying the first divergent transition. The existing safeguard is
also intentionally configurable: `SGLANG_SPEC_TP_SYNC=off` disables it. The
added boundary test demonstrates that `off` neither synchronizes nor detects a
rank-local decision.

## Reproduction evidence

The exact candidate suite passed 5/5 both before and after applying candidate
commit `5247b13d80bb3e7549c5517a7da17ca469ac8cd0`. This independently reproduces
the review's claim that the candidate is test-only hardening, not a
failing-before/passing-after runtime fix.

A two-process `torch.distributed` Gloo fixture then constructed the original
reported tensor contracts with hidden size 7168:

- rank shapes `(628, 7168)` and `(1140, 7168)`, numels 4,501,504 and 8,171,520;
- rank shapes `(6098, 7168)` and `(9682, 7168)`, numels 43,710,464 and 69,400,576.

Both `all_reduce` runs exited nonzero. Gloo reported that received data size did
not match expected size, one rank aborted with `SIGABRT`, and its peer reported
a reset connection. This reproduces the unequal-shape collective contract in a
real multiprocess process group. It does not reproduce the scheduler event
that created the row-count divergence, distributed collective ordering under
the full serving stack, or NCCL/RCCL behavior.

Raw logs and the reproduction script are retained at
`/tmp/amdpilot-repo-j-e2908ac805b9/evidence/`:

- `candidate-tests-on-base.log`
- `candidate-tests-on-candidate.log`
- `reproduce_unequal_allreduce.py`
- `unequal-628-1140-gloo.log`
- `unequal-6098-9682-gloo.log`

## Limitations

Only one AMD Instinct MI355X (`gfx950`) was available. Kimi-K3 and DSpark model
weights, eight NVIDIA B300 GPUs, CUDA 13.0, and NCCL 2.28.3 were unavailable.
Therefore full-model TP=8 multimodal chunked prefill, concurrent scheduler
transitions, semantic accuracy, the two production sequence numbers, sustained
traffic, and watchdog behavior remain unverified. The assigned GPU only ran
the retained same-device tensor test; the unequal-shape reproduction used a
two-process CPU Gloo group.
