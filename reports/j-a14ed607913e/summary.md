# Correction generation 1

Candidate: https://github.com/amdpilot-org/sglang/pull/2422 at `c89391648105d687cd6591fb2d501a7ab3c2c5a3`

Independent review: https://github.com/amdpilot-org/sglang/pull/2498

The candidate correctly expanded `ResponseToolType` to the 12 spellings accepted by the Python protocol, but its typed `ResponseTool` discarded fields it did not model when the gateway serialized the request for forwarding. The failing-before log demonstrates all three reviewed cases. The correction preserves such fields in a flattened extension map while retaining typed validation and the candidate's unknown-type rejection boundary.

Raw evidence is in this directory. No GPU, model weights, or distributed deployment was needed or used for this protocol-only defect.
