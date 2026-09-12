# Independent review of amdpilot-org/sglang PR 1611

- Candidate reviewed: `5478c5bd3941700593977490de41c1e4d177e9cf`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Upstream issue: https://github.com/sgl-project/sglang/issues/34631
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1649
- Recommendation: **request changes**
- Original boundary defect fully resolved: **yes**, in deterministic state-machine coverage

## Findings

### Blocking: rollback snapshots introduce quadratic retained memory

`ReasonerGrammarObject.accept_token()` appends `_snapshot_state()` for every
accepted token. `_snapshot_state()` makes a fresh `list(self._thinking_match_history)`.
The reasoning history is not shortened when answer generation begins, so every
answer token retains another full copy of the reasoning-length list. Retained
list slots therefore scale as `reasoning_tokens * answer_tokens`, in addition to
the per-token tuple/list overhead.

An independent 2,000-token reasoning prefix followed by 2,000 answer tokens
produced 4,001 state-history entries and 4,002,000 copied reasoning-history
slots. `sys.getsizeof()` measured 32,144,000 bytes for those 2,000 copied lists
alone, excluding tuples, integers, the first 2,001 snapshots, inner grammar
state, and application memory. Longer guided-reasoning outputs make this a
material per-request memory regression. Preserve rollback state compactly or
stop copying immutable reasoning history into every generation snapshot.

## Original-issue verification

On the exact recorded base, the issue's deterministic token stream first
invoked the JSON grammar at position 6 while the answer body began at position
11. The inner grammar received all five model-written answer-channel header
tokens.

On candidate `5478c5bd3941700593977490de41c1e4d177e9cf`, the first mask was at
position 11 and the inner grammar received only the two answer-body tokens. This
is a failing-before/passing-after reproduction of the reported contract, using
the checked-out source module rather than an installed copy.

The candidate's focused constrained/reasoning-parser suite passed: 153 tests
and 64 subtests. Independent adversarial checks passed for multi-token and
self-overlapping delimiters, a second reasoning channel, rollback and replay
across both new boundaries, a negative/unlimited malformed-header cap,
`require_reasoning=False`, immediate binding for non-channel detectors, and a
runtime inventory showing only `muse` opts into the new behavior.

## Scope and limitations

The candidate changes only Python files, so no native library or FlyDSL rebuild
was applicable. Imports were explicitly resolved to
`/job/repo/python/sglang/srt/constrained/reasoner_grammar_backend.py` at both
revisions. The host exposes one AMD Instinct MI355X (`gfx950`) through Torch
2.11.0+rocm7.2, but this defect and the reviewed tests are CPU state-machine
logic; no GPU kernel was exercised and `gpu_execution` is false.

Muse Glimmer 30B NVFP4 weights were not available. The production placeholder
rate, full HTTP serving behavior, semantic response quality, and GPT-OSS lead
remain unverified. The deterministic state-machine result supports that the
original Muse grammar-boundary mechanism is fixed, but it does not substitute
for those model-level measurements.
