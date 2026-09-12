# Multi-tokenizer batch IPC routing evidence

The checked-out base already stamped each request inside a tokenized batch, but
left the batch wrapper's `rids` and `http_worker_ipcs` unset. The failing-before
unit run captures that behavior for both generate and embedding wrappers. The
fix derives missing wrapper IDs from the items and populates the parallel IPC
list, while preserving an explicitly supplied wrapper ID list.

The GPU evidence comes from the qualified deterministic tiny Llama fixture at
commit `f1d603677ca76a9ea21124a544e405c5b0cbd315`. The runner was used with two
tokenizer workers and tokenizer batch encoding enabled; its batch probe was
changed to send two `input_ids` arrays so it exercised the native tokenized
batch path. `batch.json` contains the full request and response, and
`server.log` records the exact server arguments and the two-sequence prefill.

This is a single-GPU transport and engine-execution check with random weights,
not a model-quality, alternate-architecture, distributed, or multi-node claim.
