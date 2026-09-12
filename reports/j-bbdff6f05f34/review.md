# Independent review of PR 561 at ec65961

Recommendation: **request changes**. The candidate is a useful partial fix, but it does not fully resolve the original issue's multiple-output requirement.

## Blocking finding

`CompletionRequest.prompt` supports a list of prompts. With two prompts and `n=1`, the server produces two output choices. The candidate decides whether `sglext.request_metrics` is a list using only `request.n > 1`:

- non-streaming: `serving_completions.py` line 585 at the reviewed commit;
- streaming: `serving_completions.py` line 464 at the reviewed commit.

Consequently, the non-streaming adversarial case returned a scalar containing the first output's metrics and discarded the second output's metrics. The streaming implementation has the same selection rule. Response cardinality should follow the actual number of outputs (and preserve a stable mapping to choices), not only the number of samples requested per prompt.

## What was verified

The prepared base reproduced the missing feature with the candidate regressions copied onto the actual base implementation. At exact commit `ec65961db8684a89360088fc59216140660d76d7`, all focused candidate tests passed: one chat response test, three completion tests, and three timing-stat tests. The independent opt-out case also passed, confirming no `sglext` metrics appear by default.

The source imports came from `/job/repo/python/sglang`, using `/tmp/amdpilot-repo-j-bbdff6f05f34/venv/bin/python`. Torch 2.11.0+rocm7.2 detected one AMD Instinct MI350X with HIP 7.2.26015. No native files changed, so no native rebuild was applicable. No live model/weights path was supplied, so end-to-end HTTP generation and GPU model execution remain unverified; GPU discovery is not treated as proof of this server feature.

Raw commands and output are retained in `evidence/`, and structured claims are in `result.json`.

Upstream issue: https://github.com/sgl-project/sglang/issues/36678

Mirror issue: https://github.com/amdpilot-org/sglang/issues/562
