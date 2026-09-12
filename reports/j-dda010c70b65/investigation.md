# Streaming unknown-tool correction generation 1

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Candidate: https://github.com/amdpilot-org/sglang/pull/1595 at
`f19a67ec7f042a4c7836ca468b50e99926a76987`

Independent review: https://github.com/amdpilot-org/sglang/pull/1680

The candidate was applied unchanged before running `reproduce_review.py`. All
three review claims reproduced. Mistral returned only the first valid call and
surfaced the rest as normal text. A final chunk containing an unknown call and
a valid call left the valid call buffered for JsonArrayParser, Qwen25, Hermes,
Llama32, and Trinity; Mistral lost/flushed it. With 31-character chunks,
JsonArrayParser emitted the final valid name but left its arguments buffered.

The correction preserves the candidate's selective unknown-call skipping. It
also keeps partial separators buffered, drains all possible parser transitions
from `finish()`, and keeps Mistral canonical-array streams delegated to the
shared state machine after the initial marker is consumed.

This is a deterministic pure-Python parser correction. No GPU, model weights,
server, native build, model architecture, or distributed workload was needed
or used.
