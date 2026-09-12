# OpenAI model validation investigation

The prepared base reproduced the reported bug on both
`/v1/chat/completions` and `/v1/completions`: an explicit unknown model
returned HTTP 200 and was echoed in the response. After the patch, the same
requests return HTTP 404 with OpenAI-compatible `model_not_found` errors, while
requests naming the served model still execute successfully.

The investigation also inspected open upstream PR 31419. That candidate is not
present in this base. This patch independently verifies the behavior and adds a
stricter LoRA boundary: `base:adapter` is accepted only when the base matches
the served model and the adapter is registered.

## Evidence

- `evidence/before/`: failing-before raw requests, responses, server log, and
  process metadata.
- `evidence/after/`: passing-after equivalents from the patched source.
- `evidence/fixture-manifest.json`: deterministic fixture definition and weight
  digest.
- `create_tiny_llama.py`, `run_server_probe.py`, and
  `probe_model_validation.py`: the inspected qualification fixture adapted with
  a deterministic chat template and issue-specific probes.

The fixture uses random tiny-Llama weights and one assigned gfx950 GPU. It
qualifies HTTP transport and real engine execution only, not full-model
semantics, other architectures, or distributed serving.
