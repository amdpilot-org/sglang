# Independent review of PR 1725

Upstream issue: https://github.com/sgl-project/sglang/issues/34111

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1761

Candidate: https://github.com/amdpilot-org/sglang/pull/1725 at exact commit
`3c44c979ec2e341317e2bd1900cfe5a3548afc17`.

## Recommendation

Accept. The candidate is a production fix, not test-only hardening. It makes
the tokenizer-side `abort_sent` transition the client-visible streaming
boundary. Queued nonterminal output is suppressed after cancellation, and a
terminal scheduler abort is retained as control metadata with empty text and
empty output token IDs. A normal completion that wins the race is preserved.

The candidate's focused regression failed against recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` (two failures, including immediate
visibility of the queued `"invalid"` chunk) and passed at the candidate (three
tests). The complete tokenizer-manager unit file passed: 29 tests and 3
subtests. Independent adversarial cases also passed for repeated queued
wakeups, abort metadata preservation, a normal finish winning the race, and
unchanged non-stream behavior.

## Serving-path evidence

Both runs used the source checkout imported from
`/job/repo/python/sglang/srt/managers/tokenizer_manager.py`, one assigned AMD
Instinct MI355X (`gfx950`), ROCm 7.2, and the deterministic tiny Llama fixture
specified by PR 649 commit `f1d603677ca76a9ea21124a544e405c5b0cbd315`.

With only the production file reverted to the recorded base, the cancelled
grammar request emitted a post-cancel abort event containing `output_ids: [0]`.
At the exact candidate commit, the corresponding event contained empty text
and `output_ids: []`. The overlap peer generated 64 chunks and finished by
length in both runs.

## Scope and limitations

The reported `meta-llama/Llama-3.2-1B-Instruct` weights, CUDA 12.9, and RTX 4090
hardware were unavailable. The qualified tiny fixture has no JSON punctuation;
XGrammar therefore has no accepted JSON tokens and token 0 decodes to empty
text. The GPU contrast validates the real overlap/grammar engine and HTTP
cancellation path plus token-ID visibility, but cannot reproduce the report's
visible invalid string or qualify that model's semantic output. The source fix
is in Python streaming control logic and does not depend on model architecture
or GPU ISA, so the contract that no queued content becomes visible after the
cancellation boundary is fully addressed.

No native source changed. Native rebuild was not applicable. Torch/ROCm were
left intact. Full raw logs, events, environment data, and temporary adversarial
script were preserved outside revision switches under
`/job/review-evidence-j-c5b65889793b/`.
