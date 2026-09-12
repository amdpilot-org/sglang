# Investigation report: context-boundary NEXTN crash

Upstream issue: https://github.com/sgl-project/sglang/issues/34239

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1626

The reported H800 TP8/Qwen3.5-397B crash could not be reproduced in this job:
the assigned accelerator is one AMD MI355X (gfx950), and the reported model
weights and eight-H800 topology are unavailable.

Source inspection did identify a boundary inconsistency on the reported path.
`TokenizerManager` reserves EAGLE draft-token output slots when an explicit
`max_tokens` value is validated. When `max_tokens` is absent, however,
`Scheduler.init_req_max_new_tokens` supplies and clips the default and formerly
left only one context slot. NEXTN with the reported settings needs four slots
(`max(topk * steps, draft_tokens) = max(1 * 3, 4)`). The draft assignment path
writes those speculative positions before accepted output is length-trimmed.

The correction applies the existing shared `compute_num_reserved_tokens()`
policy to the scheduler fallback as well. With an actual sequence length of
262091, the old fallback allowed 52 more generated tokens, reaching 262143 and
leaving one slot. The corrected fallback allows 49, so four verify slots end at
262144 without crossing the configured context extent. Non-speculative decoding
retains its existing one-slot behavior.

Evidence is retained in `raw/`: the new regression fails against the old
scheduler formula and the complete focused suite passes after the change. The
GPU check confirms the boundary arithmetic and an independent tensor result on
the assigned gfx950 device; it is not a reproduction of CUDA graphs, FA3,
Qwen3.5-397B, TP8, or NVIDIA's illegal-memory-access failure.
