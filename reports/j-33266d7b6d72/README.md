# Issue 35242 investigation

The prepared base reproduced the reported validation error when a tool's
`parameters` schema contained `"required": null`. SGLang's request path runs
`normalize_json_schema_types` and then `Draft202012Validator.check_schema`;
before this change the normalizer preserved the null value and the validator
raised `None is not of type 'array'`.

This change treats only a null `required` value as the exporter sentinel for an
omitted keyword. It deliberately preserves valid arrays and invalid non-null
values so normal schema validation still rejects malformed inputs. The
regression directly invokes `OpenAIServingChat._validate_request`, with helper
tests covering a nested schema and both positive and negative boundaries.

Before implementation, the current issue and related changes were inspected.
Upstream PR https://github.com/sgl-project/sglang/pull/35631 is an open,
independent candidate for the same issue; it additionally drops
`properties: null`. The change here is narrower because the source report only
demonstrates `required: null`.

Raw test output and exit codes are under `raw/`. The source issue's attached
70,588-line server log was retained outside the worktree at
`/tmp/amdpilot-repo-j-33266d7b6d72/evidence/server.log`; it records the Qwen3.8
launch configuration and the 400 request but not the request body.

No model execution is claimed. The assigned MI350X/gfx950 was inventoried, but
GPU work is irrelevant to this CPU-side request-validation defect. The
Qwen3.8-27B-FP8 weights and full original payload were unavailable.
