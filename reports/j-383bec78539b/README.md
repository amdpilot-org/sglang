# Overlap speculative decoding intake and validation

Upstream issue: https://github.com/sgl-project/sglang/issues/11762

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2919

The issue is a roadmap, not one atomic implementation request. Current `main`
already contains implementations and registered tests for multiple boxes that
remain unchecked in the issue, including EAGLE top-k > 1 with page size > 1 and
NGRAM under the overlap scheduler. This contribution adds focused regression
coverage for NGRAM's overlap relay: accepted tokens from the prior in-flight
iteration are appended exactly once, while synchronous and grammar-barrier
paths do not append them again. It also covers malformed corpus output.

The prepared native wheel prevents an end-to-end qualification. Both the
source checkout and its Python wrapper require
`sgl_kernel::{reconstruct_indices_from_tree_mask,reconstruct_indices_from_tree_mask_cpu}`,
but pinned `sglang-kernel 0.4.6.post1` exports neither operator. A plain tiny
Llama server passed all HTTP probes; the same server with overlap NGRAM reached
the first scheduler batch and then failed at that missing native operator.
`repository-environment.json` provides no native rebuild command (`native` is
null), so no replacement wheel or toolchain was introduced.

Raw evidence is retained outside the worktree under
`/tmp/amdpilot-repo-j-383bec78539b/`: `test-logs/`, `evidence/baseline/`,
`evidence/ngram/`, the exact PR649 fixture scripts, and the generated model.
The deterministic fixture has weight SHA256
`6632fab7c351a0bd85518d80a89ec673f68139587f5ad3fa8d5ab38fad87e75f`.

Unverified roadmap portions include EAGLE model accuracy; all attention-backend
combinations; over-allocation optimization; penalty behavior; universal
`SpecTpWorker`/`TpModelWorker` compatibility; DeepEP/EP; DP multi-rank behavior;
PD disaggregation; LoRA model behavior; and separate-plan-stream performance.
The tiny random Llama fixture cannot qualify those architectures or semantics.
