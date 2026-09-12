# Independent review of PR 1192

Upstream issue: https://github.com/sgl-project/sglang/issues/37393

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1230

Candidate: https://github.com/amdpilot-org/sglang/pull/1192 at exact commit
`27b34c2785e79791a4b900633329c5863fb6c131`.

## Recommendation

Accept the candidate as test-only hardening and an honest correction of the
earlier unsupported verification claim. It does **not** fully resolve the
original issue. The candidate changes only tests and reports, and its exact six
tests pass without modification on the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` as well as on the candidate.
Consequently, it supplies no failing-before/passing-after runtime correction.

The retained tests establish that the already-present `SpecTpSync` broadcast
overwrites a same-shaped rank-local target sample, that the prefill call occurs
before publishing sequence lengths, and that `SGLANG_SPEC_TP_SYNC=off`
intentionally performs neither synchronization nor divergence detection. They
do not establish that rank-local target sampling caused either production
sequence, or that every scheduler/multimodal transition preserves identical TP
batch membership and row counts.

## Independent reproduction

The exact candidate suite was copied outside the checkout, then run against the
prepared base and candidate. Both runs passed 6/6. The `cuda` parameter ran on
the assigned AMD Instinct MI350X/gfx950 through PyTorch 2.11.0+rocm7.2. Import
checks resolved `sglang`, `spec_tp_sync.py`, and `dspark_worker_v2.py` from
`/job/repo/python`, not an installed SGLang copy.

An independently written two-process Gloo fixture constructed both original
embedding-output contracts at hidden size 7168:

- `(628, 7168)` versus `(1140, 7168)`: 4,501,504 versus 8,171,520 elements;
- `(6098, 7168)` versus `(9682, 7168)`: 43,710,464 versus 69,400,576 elements.

Both real `all_reduce` calls failed with a received-size mismatch and a worker
`SIGABRT`. This reproduces the collective contract failure, but not the serving
transition that created unequal rows and not NCCL/RCCL ordering or watchdog
behavior.

Raw evidence is retained outside the checkout at
`/job/review-evidence-j-ae2f1238509a/`, including candidate/base test logs,
environment and import paths, the independent distributed fixture and both
collective logs, the exact candidate diff, and issue/PR metadata.

## Native and architecture assessment

The candidate changes no Python runtime or native source, so no native rebuild
was applicable. The prepared environment has one AMD gfx950 GPU. It does not
provide eight NVIDIA B300 GPUs, CUDA 13.0, NCCL 2.28.3, Kimi-K3 weights, or the
DSpark draft weights. Full TP=8 multimodal chunked-prefill serving, concurrent
scheduler transitions, the recorded sequence numbers, sustained traffic, and
the NCCL watchdog outcome therefore remain unverified.

