# Independent review of candidate PR 2101

Upstream issue: https://github.com/sgl-project/sglang/issues/32486

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2060

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2135

Candidate: https://github.com/amdpilot-org/sglang/pull/2101 at
`2a536e1012cd4e2dd4254ae6009f17ea0ca4d458`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Verdict

Recommendation: **accept**.

The candidate is test-only hardening, not the production fix. The recorded
base already fully implements the original issue's pre-worker contract in
`python/sglang/srt/arg_groups/model_path_hook.py`: target, tokenizer, and
speculative draft object-store URIs are prepared, and identical paths are
deduplicated. The candidate adds direct regression coverage for that existing
behavior. No candidate prose or unrelated startup smoke was used as proof.

The original failure was independently reproduced at the issue's reported
commit `5cc273a780e1b35255717e52975c3f43ed1e9978`: the issue-shaped probe expected
target and distinct draft downloads but observed only the target and exited 1.
At the recorded base, the same contract and independent boundary cases pass.
Source history confirms that `abddb1c7e9d61ddddeaf016d885c2f20aab426e8`
introduced the production loop before the recorded base; later refactoring
moved it from `server_args.py` to `model_path_hook.py` without losing it.

## Evidence

- Historical reported-commit probe: exit 1; actual calls were
  `['s3://bucket/target-model']`, while target plus distinct draft were
  required.
- Recorded-base existing suite: 12 passed.
- Exact-candidate focused suite: 15 passed, including the candidate's three
  object-store tests.
- Independent base and candidate probe: passed explicit `runai_streamer`
  target/draft paths, all supported S3/GCS/Azure schemes (including uppercase
  S3), target/draft deduplication with a distinct tokenizer, and exclusion of
  Hub/HTTP/Redis paths.
- Candidate imports resolved to the checkout sources at
  `/job/repo/python/sglang/srt/arg_groups/model_path_hook.py` and
  `/job/repo/python/sglang/srt/utils/runai_utils.py`.
- The candidate diff contains only a Python test and report files. It changes
  no native source or built artifact, so a native rebuild is not applicable.

Raw logs and probes were preserved outside the revision-switching checkout at
`/job/review-evidence-j-8e7149364711/`.

## Scope and limitations

The prepared host exposes one AMD Instinct MI350X (gfx950) through ROCm 7.2;
the report described two nodes with TP=8, PP=2, and NVIDIA H20 GPUs. No object
store credentials or model repositories were provided. Therefore this review
does not claim a live S3-compatible transfer, distributed serving, weight
streaming, model execution, or semantic-accuracy reproduction. Those are not
needed to verify the narrow pre-worker URI preparation/call-count contract,
which was exercised at the actual handler boundary. No GPU kernel was run for
this CPU control-flow issue.

No remaining counterexample was found within the original contract. Live
network and distributed deployment behavior remain unverified environment
limitations rather than evidence against the candidate.
