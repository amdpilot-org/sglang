# Independent review of amdpilot-org/sglang PR 1063

- Upstream issue: https://github.com/sgl-project/sglang/issues/37186
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1102
- Candidate: https://github.com/amdpilot-org/sglang/pull/1063
- Exact candidate commit: `bbaaba6517fbc5bc7c2ff6663e50f93fc9ddf1f9`
- Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**

## Finding

The candidate fully resolves the original name-filtering defect in the implementations under review. On the recorded base, an actual `Qwen3_5ForConditionalGeneration.load_weights` fixture with PP range `[1, 3)` dropped `model.visual.layers.3.weight` and `model.visual.layers.4.weight`; only decoder layers 1 and 2 were loaded. The reproduction exited 1 on the first required visual weight.

At the exact candidate commit, the repository regression passed (7 tests), and an independent adversarial fixture passed for both `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`. Decoder layers 0 and 3 were excluded, decoder layers 1 and 2 loaded, and visual encoder layers 0, 3, and 5 all loaded. Namespace boundary checks also confirmed that only normalized `model.layers.N.` names receive the decoder PP filter; visual, audio, embedded, prefixed, and near-match namespaces do not.

The implementation is appropriately narrow: after the existing checkpoint-name normalization, `_get_decoder_layer_id` recognizes only `model.layers.`. It therefore retains PP filtering for decoder weights while preventing the broad `layers\.(\d+)\.` utility regex from classifying encoder-internal layers as decoder layers.

## Source and native-path verification

The tested module imported from `/job/repo/python/sglang/srt/models/qwen3_5.py`, proving the checkout source rather than an installed copy was exercised. The candidate changes Python source, its unit test, and report artifacts only. It changes no C++, HIP, CUDA, FlyDSL, or other native source, so no native rebuild was applicable (`native_rebuilt: false`).

## Environment and scope limitations

The prepared interpreter used PyTorch `2.11.0+rocm7.2` with HIP `7.2.26015`. One assigned `gfx950` GPU was visible as AMD Instinct MI350X. The original report used NVIDIA H100/CUDA, but this defect is deterministic Python weight-name filtering before device execution; the tests intentionally did not claim GPU execution.

No Qwen3.5 checkpoint weights were available, and no complete model, multi-process PP server, or multi-node workload was run. The actual loader implementations were exercised with minimal deterministic modules, which qualifies the filtering contract but not semantic model accuracy or end-to-end serving. No remaining counterexample was found within the original contract.
