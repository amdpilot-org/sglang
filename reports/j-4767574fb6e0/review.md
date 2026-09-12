# Independent review of PR 773

Reviewed exact commit `4a7bd3437e120c2cd10a66c0e8fae4540362b814` against base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept**, with `fully_resolves_original: false` because full target/drafter serving and acceptance were not available to measure.

The base failure was reproduced through the actual `Qwen4ExpModel.forward`: selected layer flags yielded only `(hidden_states, hc_hidden_states)`, with the second tensor still HC-wide and no auxiliary list. The candidate's capture occurs before selected internal layer `k`; the public setter maps requested target layer `k-1` to internal layer `k`, matching the parent communicator's completed-previous-layer tap order.

The candidate does more than repair width. It invokes the selected layer's learned `attn_hyper_connection.mix`. On the assigned MI355X, the real implementation matched an independently expanded per-branch RMS normalization, learned low-rank gate, and reduction to `2.98e-08` maximum absolute error. It differed from a semantically incorrect uniform branch average by `1.8565`.

Ordinary non-idle return arity, idle tensor return, HC consumer routing, auxiliary logits routing, and public layer-ID shifting passed focused checks. The source imported from `/job/repo/python/sglang/srt/models/qwen4_exp.py`. No native files changed and no native rebuild was applicable.

Raw command output was retained outside the revision-switching checkout in `/job/review-evidence-j-4767574fb6e0/`; the concise evidence record is in `raw-output.txt`.

Unverified: real Qwen3.8-Flash-Next plus DFlash/DSpark checkpoint loading, generation, draft acceptance, TP/PP/distributed execution, PLE/MoE/attention integration, graph capture, and the separate `sample_from_anchor` layout concern.
