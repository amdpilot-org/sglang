# Independent review of PR 2001

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/2001 at exact commit `90b5ae30848c43a54f1ad293ab3b4dd2ce4bf478`

Upstream issue: https://github.com/sgl-project/sglang/issues/33164

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2038

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/1945

## Finding

Recommendation: **request changes**. The candidate fixes the reported production-value examples and ordinary list, tuple, and nested cases, but does not fully satisfy the original stated contract for all `max_length` values.

On the recorded base, with 10,000-character strings and `max_length=8`, the JSON result grows from 120 bytes for eight elements to 80,039 bytes for nine elements, while both text branches emit about 80 KB. At the exact candidate, the corresponding outputs are 127 and 159 bytes, nested collections are recursively truncated, the candidate's four tests pass, and its million-character/`max_length=2048` fixture produces the claimed bounded results.

The independent `max_length=1` case still fails because `half_length` becomes zero and Python's `data[-0:]` is the full sequence. For two 10,000-character strings, JSON returns three items and retains a string of length 10,003; text returns 20,029 characters. This violates the issue's expectation that `max_length` bound both collection element count and each element's size in every branch. The zero-half string-tail behavior predates the candidate, but the candidate's new recursive path does not account for it and therefore the proposed full-fix claim is too broad.

## Validation and environment

The prepared checkout exactly matched recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; the candidate's parent is that same commit. Both revisions imported `sglang` and `request_logger.py` from `/job/repo/python`, using `/tmp/amdpilot-repo-j-030ca4a13f04/venv/bin/python`.

No native sources changed, so no native rebuild applies. No GPU execution was used because these are pure-Python serialization helpers. The environment exposed one AMD Instinct MI350X (gfx950 family), ROCm 7.2, and Torch 2.11.0+rocm7.2, but GPU execution would not exercise the affected code. No server or model test was needed for the same reason.

Raw outputs are retained in `raw/`. The candidate also contains trailing whitespace in two report artifacts, so a diff-wide `git diff --check` is not clean; this is not the reason for the functional recommendation.
