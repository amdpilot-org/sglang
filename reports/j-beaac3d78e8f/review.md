# Independent review of candidate PR 581

Candidate: `a029c210fb032ae8127b6113496753b7b057bd1a`

Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fully resolves the original issue.

The base implementation reproduced the report exactly. A producer batch with
one legacy seven-element `BlockStored` and one salted eight-element
`BlockStored` decoded through `KVEventBatch` as two plain `BlockStored`
instances and discarded the salt. Decoding both as
`BlockStoredWithMetadata` rejected the unsalted event, while placing both
producer structs in a union failed because their tags collide.

At the exact candidate commit, the new exported decoder-side
`BlockStoredView` accepted both wire lengths and exposed `cache_salt` as the
original string or `None`. `KVEventBatchView` also preserved the other event
kinds and batch rank. The producer classes and their seven/eight-element wire
shapes were unchanged.

Independent cases covered exact hand-built legacy bytes, salted and unsalted
producer instances, Unicode and embedded-NUL salt data, all event variants in
one batch, explicit null metadata, unknown future metadata keys, malformed
metadata, and the documented warning that `BlockStoredView` is decode-only.
No counterexample remained.

Raw command output and standalone reproduction scripts are preserved outside
the revision-switching checkout at
`/job/review-evidence-j-beaac3d78e8f/`.

Architecture and environment: the prepared interpreter was
`/tmp/amdpilot-repo-j-beaac3d78e8f/venv/bin/python`; imports resolved to the
repository source at `/job/repo/python`. This contract is CPU-only msgspec
serialization, so GPU execution was neither required nor performed. The
candidate contains no C++ or native changes, `native` is null in the prepared
environment metadata, and no native rebuild was applicable. A live server and
ZeroMQ transport were not started; the exact producer-generated msgpack bytes
at the reported schema boundary were tested directly.
