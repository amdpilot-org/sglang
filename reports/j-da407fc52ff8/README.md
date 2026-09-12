# `/v1/responses` nullable stream regression

The prepared base reproduced the reported failure when a gateway-shaped request
contained `"stream": null`: `ResponsesRequest` accepted the value, then
`OpenAIServingResponses._make_request` passed it to the strict boolean
`ChatCompletionRequest.stream` field and Pydantic raised `bool_type`.

The source fix normalizes only that adapter value from `None` to `False`.
The regression independently covers `None`, explicit `False`, explicit `True`,
and omission (whose protocol default was already `False`).

Raw evidence is under `raw/`. `failing_before.txt` contains the exact pre-fix
validation traceback. `passing_after_focused.txt` and
`passing_after_suite.txt` contain the passing unit results. The successful
source-checkout GPU server run is under `raw/e2e_llama2/`, including every
request/response, the full server log, and cleanup metadata.

The end-to-end fixture is a deterministic two-layer random Llama and validates
only HTTP transport plus engine execution. It does not reproduce the reported
DeepSeek-V4 architecture, model semantics, gateway process, or distributed
configuration.
