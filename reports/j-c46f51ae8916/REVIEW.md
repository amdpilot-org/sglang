# Independent review of candidate PR 664

Reviewed exact commit `183f57d6899bbb1df7a273383fa779ad94bffae7` against the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **request changes**. The generic eager boundary correctly fails closed and the DSA attention-TP padding correction is valid on the assigned AMD GPU. The DSA opt-in contract is incomplete, however: it validates only three DSA arrays plus `out_cache_loc`, then permits model execution. An independent case using the real `ForwardBatch` and `DeepseekSparseAttnBackend` validator was accepted with four physical tokens but three-row `dsa_seqlens_expanded` and `token_to_batch_idx`. Both are row-indexed DSA metadata used downstream. Thus the candidate does not establish the original invariant that every row-indexed buffer covers the declared execution extent.

Raw observations are in `raw/`; structured claims and limitations are in `result.json`.
