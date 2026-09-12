# SGLang ROCm environment qualification: j-1fc9870820b9

Status: **qualification pending**

This report qualifies the prepared SGLang checkout and Python environment on its
assigned ROCm GPU. It is an environment/platform exercise, not a claim that any
upstream model-quality issue is resolved.

Platform context: https://github.com/amdpilot-org/amdpilotv2/pull/435

Planned coverage:

- create a deterministic, tiny local Llama fixture and tokenizer;
- launch the real SGLang server on the assigned GPU using a ROCm-supported
  attention backend;
- probe `/generate`, `/v1/completions`, batching, streaming, and a small graph
  capture configuration where supported;
- retain server logs and exact request/response evidence, record any repairs,
  verify cleanup, and publish a machine-readable `result.json`.

The final qualification outcome and reproduction instructions will replace this
pending status in the same pull request.
