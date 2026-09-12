# Qualification of Rust `/server_info.kv_events`

Candidate `760ef21de6b20abcf4ee851c4e895d1c9467d512` was checked out temporarily over prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`. The candidate is verified for the requested environment, native build, Rust test, route, typed-state, redaction, malformed-input, IPv6, and Python-parity checks.

The key route evidence is `evidence/axum-route-contract.txt`. The exact temporary test source is in `evidence/temporary-route-test.patch`; it sends real Axum requests to the production route, receives the emitted control request, replies through its real response sink, and checks the resulting HTTP JSON. It was reverted before returning to the delivery branch.

The forced native rebuild is recorded in `evidence/native-rebuild.txt`, including the imported path, file size, inode, mtime, and SHA-256. Full commands and measurements are summarized in `result.json`; raw outputs and candidate/issue snapshots are retained under `evidence/`.
