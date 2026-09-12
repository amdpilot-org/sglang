# Independent review of PR 3143

Candidate: https://github.com/amdpilot-org/sglang/pull/3143  
Exact commit: `07dba2d6a075964c1d5845942b8c4ddb748f75d0`  
Upstream issue: https://github.com/sgl-project/sglang/issues/31021  
Mirror issue: https://github.com/amdpilot-org/sglang/issues/3149

## Recommendation

**Request changes.** The candidate is a meaningful partial implementation and
does fix the two exact counterexamples carried forward from PR 3059: deleted
definitions no longer remain on the reloaded module, and a preserved class can
change from `BaseA` to `BaseB`. Its eight focused tests pass at the exact commit.

However, deletion is still incomplete for the candidate's stated direct-import
rebinding contract. If another loaded SGLang module previously executed
`from selected_module import removed`, deleting `removed` from the selected
module leaves that consumer alias callable with the stale implementation.
`reload_modules` reports success. The implementation only creates replacement
mappings when a same-named new function/class exists, so it cannot repair or
invalidate aliases to deleted definitions.

## Reproduction

The prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` has neither
`python/sglang/srt/dev_reload.py` nor the `/dev/reload` route, reproducing the
original absence of the requested feature.

At candidate commit `07dba2d6a075964c1d5845942b8c4ddb748f75d0`:

```bash
/tmp/amdpilot-repo-j-39dbd62698b8/venv/bin/python -m pytest -q test/srt/test_dev_reload.py
# 8 passed, 17 warnings
```

The independent deletion/consumer case produced:

```text
reload_success ('sglang.delete_source',)
source_removed_absent True
consumer_stale_alias_callable stale
```

The candidate's prior-counterexample script independently produced deletion of
the source-module names, preservation of class identity, a `BaseB` base, and
correct results from both existing and new instances.

## Scope and limitations

No native source changes exist, so a native rebuild is not applicable. The
prepared interpreter imported candidate code from `/job/repo/python/sglang`.
This review did not rerun the candidate's retained MI350X tiny-Llama serving
session; therefore it makes no independent GPU execution claim. Multi-rank
coordination, large-model behavior, other model architectures, torch.compile,
JIT/native changes, and failure recovery remain unverified. Reload is explicitly
non-transactional. The optional fault-tolerant serving loop is not implemented.
