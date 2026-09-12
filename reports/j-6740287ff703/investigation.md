# Investigation: object-store speculative draft metadata

Upstream issue: https://github.com/sgl-project/sglang/issues/32486

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2060

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The prepared base already contains the solution described by the open issue.
At the reported commit, `ServerArgs._handle_model_source_paths()` downloaded the
target URI and a distinct tokenizer URI, but never inspected
`speculative_draft_model_path`. An issue-shaped probe against `5cc273a780`
therefore observed only:

```text
['s3://bucket/target-model']
```

when the required calls were target plus draft.

Commit `abddb1c7e9d61ddddeaf016d885c2f20aab426e8` (upstream PR #32541,
"[Kimi] Support kimi-k3") changed that handler to iterate over target,
tokenizer, and speculative draft paths, using a set to download each unique
object-store URI once. Later configuration refactors moved the same logic to
`python/sglang/srt/arg_groups/model_path_hook.py`; it remains present at the
recorded base.

## Added regression

`TestTheObjectStoreArm` directly pins the reported contract:

- distinct target, tokenizer, and explicitly `runai_streamer` draft URIs are
  all prepared before startup;
- a shared target/tokenizer/draft URI is prepared exactly once;
- ordinary Hub paths and other remote connector schemes are not sent to the
  RunAI object-store downloader.

The focused source-path suite passes all 15 tests. Raw command output is kept
under `/tmp/amdpilot-repo-j-6740287ff703/evidence/`.

## Scope

This is pre-worker control flow and does not execute a GPU kernel. No real S3
credentials or model repositories were supplied, and the original H20
two-node TP=8 PP=2 deployment was not recreated. The evidence qualifies the
metadata-preparation call contract only, not network transfer, distributed
serving, model execution, or semantic accuracy.
