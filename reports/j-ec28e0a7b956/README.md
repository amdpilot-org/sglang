# Independent review of PR 1520

Upstream issue: https://github.com/sgl-project/sglang/issues/34974

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1554

Candidate: https://github.com/amdpilot-org/sglang/pull/1520 at
`ad7df07c3e50608a589021ee0b2897c77133cef2`

Recommendation: **accept**.

At the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, an
independent fixture invoking the checked-out `DSparkV4Stage._run_ffn` and the
real EPLB stat gatherer reproduced the issue's exact failure:
`RuntimeError: Index tensor must have the same number of dimensions as self tensor`.
The import path was `/job/repo/python/sglang/srt/models/deepseek_v4_dspark.py`.

I then detached the prepared checkout at the exact candidate commit. Its
three-test regression passed, and an independent adversarial fixture confirmed
that repeated draft calls suppress recorder hooks even for invalid draft expert
IDs, preserve an enclosing target layer index, restore recorder state, and do
not alter output when recording and capture are inactive. A real HIP graph was
captured and replayed on the assigned AMD Instinct MI350X (gfx950); output was
exact and target counts remained unchanged. Imports continued to resolve from
the checkout. The candidate changes Python only, so no native rebuild was
applicable.

The patch follows the existing NextN/MTP ownership rule: draft MoE selections
must not be attributed to the target model's EPLB distribution. Disabling the
shared target recorder only around the DSpark draft MoE call removes the
`layer_idx=None` failure without inventing a target layer or corrupting target
statistics. No remaining counterexample to the reported crash contract was
found.

The original DeepSeek-V4-Flash weights and 8 NVIDIA H20 GPUs were unavailable.
Consequently this review does not claim a full-model TP=8/EP=8 server startup,
NVIDIA CUDA graph, semantic-accuracy, speculative-acceptance, or live EPLB
migration run. The single-gfx950 seam test validates the actual Python call
path, recorder behavior, and graph capture/replay, not the unavailable
deployment topology.
