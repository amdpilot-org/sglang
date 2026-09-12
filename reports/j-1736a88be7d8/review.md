# Independent review of amdpilot-org/sglang PR 1564

Candidate reviewed: `5e007ae4fb75b5b0b68c0c37a7ffe913239ca402`

Upstream issue: https://github.com/sgl-project/sglang/issues/34740

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1598

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1564

## Recommendation

Request changes. The candidate is a verified partial fix for Defect B under the
reported fixed-size speculative chunks, but it does not fully resolve the
original issue because Defect A remains unchanged. On the assigned gfx950 GPU,
the exact candidate still makes the fixed-token simulation path overwrite every
prediction with token id 100.

## Evidence

The prepared checkout was exactly the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. Running the candidate's regression
against that base produced `1 failed, 2 passed`: a repeated byte-fallback stream
never committed text. At the exact candidate commit, the submitted regression
plus the existing stop-trimming tests produced `11 passed`.

The interpreter imported `sglang` from `/job/repo/python/sglang/__init__.py` and
the changed manager from
`/job/repo/python/sglang/srt/managers/detokenizer_manager.py`. The candidate
changes Python only, repository-environment.json declares no native build
target, and no native rebuild was required.

Independent adversarial checks found:

- Repeated U+FFFD-producing chunks of one or four tokens retained a live window
  independent of a 32-step versus 256-step response.
- Tokens decoding to the empty string also retained a response-length-independent
  window.
- The bound counts decode events, not tokens. With 4,096 invalid byte tokens per
  event, the retained live window was 57,344 tokens after 32 events and 53,248
  after 256 events. The window is therefore bounded for fixed-size events but is
  not capped to a small token tail regardless of event size.
- On one AMD Instinct MI355X (`gfx950:sramecc+:xnack-`) with ROCm 7.2, the actual
  `generate_simulated_accept_index` path returned acceptance indices
  `[[3, 4, 5, 6]]`, `num_correct_drafts=[3]`, and `predict_unique=[100]`.

## Scope and limitations

DeepSeek-V4-Pro weights and the reported eight-GPU TP/DP/EAGLE deployment were
not available. This review therefore does not claim the reported throughput,
TTFT, semantic output, tokenizer vocabulary, or distributed serving workload
was reproduced. The GPU check qualifies the actual simulated-acceptance tensor
path only. The deterministic tokenizer fixture qualifies the real Python
detokenizer control flow, not DeepSeek-V4-Pro.

Raw command output and fetched issue/PR metadata were retained outside the
checkout under `/job/review-evidence/j-1736a88be7d8/` while revisions were
switched.
