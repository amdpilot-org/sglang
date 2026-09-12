# Investigation evidence

The recorded base still contained the reported defect. `release_kv_cache()` passed `Req.effective_kv_committed_len()` as `kv_len_to_handle`, while both external backends recomputed a longer store length and sliced token IDs and KV indices using that recomputed value.

The failing-before regression called the actual `LMCRadixCache.cache_finished_req` and `FlexKVRadixCache.cache_finished_req` implementations with optional-package stubs. With a caller boundary of four, normal mode sent nine tokens/slots and speculative top-k mode sent eight. The fix applies the caller value only as an upper bound, after each backend's existing committed-length calculation.

Related work was checked before implementation. Upstream issue https://github.com/sgl-project/sglang/issues/34539 remains open, and open upstream PR https://github.com/sgl-project/sglang/pull/34541 proposes the same narrow source correction. The prepared base did not contain it.

Raw commands and outputs are retained in `reports/j-22d2b5550954/raw/`.
