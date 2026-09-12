# Independent review of amdpilot-org/sglang PR 1872

Candidate: https://github.com/amdpilot-org/sglang/pull/1872

Exact commit: `b52daa1dd1564889b9eb2e59ef8c6bfcd7e32091`

Upstream issue: https://github.com/sgl-project/sglang/issues/34631

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1877

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Recommendation

Request changes. The candidate fixes the reported Muse grammar-arming boundary and the independently reported quadratic snapshot storage, but introduces a rollback contract regression: `ReasonerGrammarObject.rollback(0)` mutates the inner grammar.

## Reproduction and validation

All Python commands used the prepared interpreter `/tmp/amdpilot-repo-j-d4e1bc44ae38/venv/bin/python` with `PYTHONPATH=/job/repo/python`. The probes recorded `/job/repo/python/sglang/srt/constrained/reasoner_grammar_backend.py` as the imported implementation.

On the prepared base, the issue's model-free stream first applied the grammar mask at position 6, immediately after `<|eom|>`, while the answer body began at position 11. The inner grammar incorrectly accepted all five answer-header tokens.

At exact candidate commit, the first constrained position was 11 and the inner grammar accepted only the two answer-body tokens. The 2,000-token reasoning plus 2,000-token answer measurement retained 4,006 scalar snapshots (the extra five are the Muse answer header), zero nested history lists, and zero copied history slots. Rolling back the answer, header, and terminator restored the 2,000-token reasoning state and an empty inner grammar.

The candidate's focused suite passed: 155 tests and 64 subtests. Independent replay-oracle checks passed for every positive rollback depth in a deterministic stream with overlapping multi-token markers and two reasoning channels, plus 500 randomized streams spanning header caps 1, 2, 4, 16, and unlimited.

## Remaining counterexample

The new expression `self._state_history[-k:]` is incorrect for `k == 0`, because `-0` is `0` and the slice therefore contains the entire history. `steps_after` counts all generation snapshots and calls `self.grammar.rollback(steps_after)`, while the following state rollback loop executes zero times.

Observed on the same simple stream:

```text
prepared base: before=[20, 21], after rollback(0)=[20, 21], tokens_after_end=2
candidate:     before=[20, 21], after rollback(0)=[],       tokens_after_end=2
```

This leaves the wrapper and inner grammar inconsistent. The candidate lacks a zero-depth rollback regression test.

## Scope and limitations

No native source changed, so no native rebuild was applicable. These state-machine and memory tests are CPU-only; no GPU execution was needed or performed. Muse Glimmer 30B NVFP4 weights were unavailable, so the production placeholder rate, semantic output quality, and full HTTP serving path remain unverified. GPT-OSS behavior also remains unverified. The deterministic tests establish the token-boundary correction, but cannot qualify model-specific semantic behavior.
