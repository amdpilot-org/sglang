# Independent review of amdpilot-org/sglang PR 730

Reviewed exact candidate commit `155602773824a492fd25629157d8da4e59508413` against prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the original issue contract.

Recommendation: **accept**. The candidate fully resolves the original CPU request-path issue within the tested scope. No remaining counterexample was found.

## Findings

On the prepared base, an independent probe reproduced the original loss: omitted chat sampler fields were eagerly materialized as OpenAI/model defaults, and the tokenizer-manager merge therefore lost preferred values. For example, preferred `temperature=0.7`, `top_p=0.8`, `top_k=20`, and `presence_penalty=1.5` became `1.0`, `1.0`, `-1`, and `0.0` respectively.

The same base probe reproduced the constraint-family problem: explicit text and explicit regex/EBNF/JSON-schema cases could retain mutually exclusive preferred constraints and fail `SamplingParams.verify`.

At the exact candidate commit, the independent probe confirmed:

- omitted chat fields use server preferences, including the five `get_param` fields and penalties;
- explicit request values retain precedence;
- generation-config values remain when no corresponding preference exists;
- a tool constraint derived from explicit tools plus `tool_choice=required` is preserved over a preferred regex;
- explicit `response_format={"type":"text"}` removes preferred JSON-schema/structural-tag constraints;
- explicit regex, EBNF, and JSON-schema constraints remove conflicting preferred constraint kinds and pass `SamplingParams.verify`;
- requests without explicit-key metadata retain the legacy native-request precedence behavior.

The imported runtime modules were confirmed to come from `/job/repo/python/sglang/...`, not an installed copy. The candidate changes only Python request/manager code and tests; no native rebuild was applicable.

## Evidence

Raw outputs are retained in `evidence/`:

- `base_probe.txt`
- `candidate_probe.txt`
- `candidate_import_paths.txt`
- `candidate_regression.txt`
- `candidate_related_tests.txt`
- `upstream_issue.json`
- `candidate_pr.json`

The candidate regression suite passed 8 tests plus 3 subtests. The broader OpenAI protocol and serving-chat suites passed 178 tests plus 93 subtests. Changed runtime Python files compiled, and `git diff --check` passed.

## Scope limits

This is CPU request-path validation. No GPU, model weights, live HTTP server, semantic generation, alternate model architecture, or distributed workload was needed or exercised. The review therefore makes no claim about those areas. No C/C++ or other native source changed.
