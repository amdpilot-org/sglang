# Issue 35498 investigation

The prepared base already implements the issue's performance-preserving fix.
`Engine._serialize_tensors_per_rank` creates an independent
`MultiprocessingSerializer` payload for each TP rank, and
`TpModelWorker._deserialize_own_rank` consumes only the entry for its rank.

`evidence/legacy_one_payload.txt` records the original one-payload/eight-consumer
reproduction: one consumer succeeds and seven fail with `EOFError`, accompanied
by seven `resource_sharer` `KeyError` tracebacks. The added registered test uses
the current Engine helper with four real spawned consumers and records all three
passing cases in `evidence/regression_test.txt`.

The available single gfx950 cannot run the reported TP8 server configuration.
No model weights were used, so this result deliberately does not claim an HTTP,
full-model, semantic-accuracy, or multi-node reproduction.
