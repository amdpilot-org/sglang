# DSpark draft LoRA investigation

The base implementation explicitly initialized LoRA only when
`not self.is_draft_worker`, so a DSpark draft worker could not own or route an
adapter. This change adds a separate DSpark draft manager selected by
`--speculative-dspark-lora-path`. The adapter is pinned, kept independent from
target-model adapters, and its deterministic internal ID is attached to each
request row before the draft forward.

This contribution is intentionally reported as `candidate_verified`, not as a
complete solution to the feature request. It supplies one adapter for the whole
server. The issue's broader goal of choosing different draft adapters for
different tasks still needs a request-level draft-adapter field, tokenizer-side
registry/accounting, scheduler admission and draining, dynamic endpoints, and
mixed-adapter CUDA-graph coverage.

Raw test logs are retained in `raw/`. The pinned interpreter had neither
`ruff` nor `black`; `git diff --check` passed. No C++ or native extension was
changed, so a native rebuild was not applicable.
