# Independent review of abort cleanup candidate

Reviewed https://github.com/amdpilot-org/sglang/pull/2096 at exact commit
`82684a5c845f51c6a20cac3cfb4fee22afef066a` against
https://github.com/sgl-project/sglang/issues/34113.

Recommendation: **request changes**. The candidate is a valid narrow fix for
pre-normalization streaming-response cleanup, but it does not fully resolve or
verify the original `/abort_request` issue.

The candidate regression was independently preserved outside the checkout. On
the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, three of its four
cleanup cases fail with `AttributeError: 'GenerateReqInput' object has no
attribute 'is_single'`. At the exact candidate, all four pass. This proves the
one-line source change fixes the delayed `create_abort_task` cleanup for
unnormalized single requests and handles list/empty-list boundaries.

However, the public endpoint is unchanged. `POST /abort_request` calls
`TokenizerManager.abort_request` directly and returns 200 after local dispatch;
it does not call `create_abort_task`, wait for scheduler cancellation, or
propagate a later cancellation failure. Therefore the candidate's passing test
does not demonstrate the original endpoint contract.

Independent live probes used the qualified deterministic tiny Llama fixture
from PR649. On both base and candidate, the endpoint returned 200 and the live
stream ended with `finish_reason.type=abort` after four generated tokens. No
`is_single` exception appeared in either server log. Thus the original failure
was not reproduced in this environment, and the candidate cannot be credited
as a full issue fix.

## Reproduction

Use `/tmp/amdpilot-repo-j-9555abf989bc/venv/bin/python`:

```bash
python -m pytest -q /tmp/amdpilot-repo-j-9555abf989bc/review-evidence/test_candidate.py -k TestCreateAbortTask
python -m pytest -q test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py -k TestCreateAbortTask
```

The first command was run on the recorded base and the second on the exact
candidate. Live commands and complete request/server evidence are retained in
`evidence/base` and `evidence/candidate`.

## Limitations

The reported Llama-3.2-1B-Instruct weights and RTX 4090/CUDA environment were
unavailable. Live execution used one AMD Instinct MI350X (`gfx950`) with ROCm
7.2 and a tiny deterministic random Llama. This qualifies HTTP transport and
real engine execution only, not the original model, NVIDIA architecture,
semantic output, or a distributed workload. No native source changed, so a
native rebuild was not applicable. Both server process groups required forced
cleanup after exceeding the runner's graceful shutdown timeout.
