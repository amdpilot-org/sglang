# Llama multi-layer EAGLE startup investigation

Both failures from the source issue were reproduced on the prepared checkout and
assigned gfx950 GPU with a deterministic, private two-layer random Llama fixture.
The raw serving logs are retained in `evidence/`.

The implementation's multi-layer worker creates one model runner per embedded
MTP layer and passes `draft_model_idx` to its model class. Ordinary
`LlamaForCausalLM` has neither embedded MTP layers nor that constructor argument.
The narrow correction therefore rejects this invalid combination during argument
resolution, before loading target weights, rather than changing ordinary EAGLE's
existing Llama tree defaults or suppressing the loader keyword.

The fixture is not a semantic or full-model reproduction. It qualifies the
reported startup path and GPU model loading only.
