# Investigation: dynamic LoRA with multiple tokenizer workers

Upstream issue: https://github.com/sgl-project/sglang/issues/31084

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2308

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Evidence

The prepared source still constructed `LoRARegistry` and `lora_ref_cache`
inside each `TokenizerManager`, while the dynamic load/unload handlers updated
only the manager serving the request. `MultiTokenizerRouter` had no LoRA update
broadcast or acknowledgement path.

The regression was run before the production change. Both dynamic load APIs
returned success with `tokenizer_worker_num=2`, proving that the selected
worker accepted and locally registered the adapter. Unload reached the local
empty registry instead of rejecting the unsupported configuration. Four tests
failed; see `raw/pytest_before.log`.

Two related upstream pull requests were inspected before implementation:

- https://github.com/sgl-project/sglang/pull/31724 remains open and proposes a
  fail-fast guard.
- https://github.com/sgl-project/sglang/pull/36487 proposed the same direction
  but is closed and is not present in the prepared base.

## Correction

All three dynamic LoRA entry points now reject `tokenizer_worker_num > 1`
before contacting the backend or changing tokenizer-local state. The message
directs users to a single tokenizer worker, or to startup preloading for
adapters stored on disk.

This deliberately does not claim multi-worker dynamic LoRA support. Correct
synchronization would require shared ID assignment, an acknowledgement barrier,
partial-failure handling, and consistent eviction semantics across workers.

## Limitations

No model weights were available or needed for the deterministic tokenizer
control-plane reproduction. No HTTP server or GPU model execution was run, so
transport behavior with a real model was not independently qualified. The
tests execute the actual handler and registry implementations with mocked
backend communication; they do not validate model architecture behavior,
semantic accuracy, or a distributed workload.
