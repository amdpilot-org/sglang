# Independent review of PR 731

Candidate: https://github.com/amdpilot-org/sglang/pull/731 at exact commit `59cbe21356107b9010878ffe686d600fe25de3ac`

Upstream issue: https://github.com/sgl-project/sglang/issues/38587

Mirror issue: https://github.com/amdpilot-org/sglang/issues/766

## Finding

The candidate is an honest partial fix and test-only hardening, not a full JSON Schema semantics fix. Its xgrammar dependency upgrade from 0.2.1 to 0.2.2 rejects the original invalid named-object value while preserving valid named integers and valid additional-only objects. It also adds a passing regression for that behavior.

The independently reported counterexample remains: `{"extra":{"data":"ok"},"data":2}` is valid under the schema but is rejected, while the equivalent named-first order is accepted. The candidate records this as a strict expected failure, so its passing suite is not evidence that the counterexample is fixed.

Direct `xgrammar.Grammar.from_json_schema` checks reproduce the ordering defect in xgrammar 0.2.2 and 0.2.6 without SGLang's Kimi-K3 structural-tag builder. On the available evidence, changing SGLang source would be speculative. The candidate appropriately preserves the partial dependency fix and documents the external compiler limitation.

## Recommendation

Accept as a bounded partial fix and regression-hardening deliverable. Do not describe it as fully resolving the original JSON Schema contract. A complete resolution still requires an xgrammar compiler version that accepts schema-valid object members in either order while rejecting invalid named-property values.

## Environment and scope

Tests used the prepared interpreter `/tmp/amdpilot-repo-j-769325de1fc7/venv/bin/python` and source imports from `/job/repo/python`. Candidate compiler tests explicitly imported private xgrammar 0.2.2 and 0.2.6 installations under `/tmp/amdpilot-repo-j-769325de1fc7/`; paths and versions are captured in raw output.

The host is x86_64 with an AMD EPYC 9965 and one visible AMD Instinct MI350X using Torch 2.11.0+rocm7.2 / HIP 7.2. No GPU execution was needed: the issue is deterministically reproduced in the CPU grammar compiler. No native source changed, the prepared environment declares no native artifact, and no native rebuild was applicable. No Kimi-K3 weights or HTTP serving path were exercised; those cannot add evidence to the deterministic compiler contract evaluated here.
