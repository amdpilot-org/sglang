# Investigation evidence

- Upstream issue: https://github.com/sgl-project/sglang/issues/34205
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1637
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Source under test: `python/sglang/srt/managers/tokenizer_manager.py`
- Regression suite: `test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py`

The prepared base still acquired request-level LoRA usage in
`_resolve_lora_path()` and released it on normal terminal batch output, but
`_handle_abort_req()` deleted the request state without a release. The retained
failing-before JUnit report records an await count of zero for the direct abort.

Related upstream work was checked before implementation: open PRs #34215 and
#37824, closed PR #34225, and closed earlier PR #29565. Review history on these
changes identified the 500/503 producer/consumer double-release boundary. The
implemented ownership check preserves the existing consumer fallback when it
still owns the exact `ReqState`, while preventing a second release after the
direct abort handler already removed and released that state.

JUnit evidence is retained under `reports/j-cbbfec46d0a2/evidence/`.
