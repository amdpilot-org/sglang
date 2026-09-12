# Independent review of PR 3367

Candidate: https://github.com/amdpilot-org/sglang/pull/3367 at `0b4099988e6bd675e3ff4ab159afa37eaeba3b2f`

Upstream issue: https://github.com/sgl-project/sglang/issues/38580

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3371

Recommendation: request changes. The candidate is a partial fix, not a complete enforcement of the original contract.

The recorded base reproduces the planner mismatch: with three real rows and attention TP size two, `cal_padded_tokens` and `pad_dsa_cache_seqlens` remain at three although the physical model extent becomes four. The candidate corrects those helpers and its 14 focused tests pass on the prepared MI350X environment.

An independent adversarial case still violates the proposed invariant. `ForwardBatch._pad_inputs_to_size` pads `out_cache_loc` with the generic zero fill. The DSA extend forward subsequently passes the complete physical `out_cache_loc` and `k` tensors to `set_mla_kv_buffer`. The candidate's validator only checks `len(out_cache_loc) == physical_tokens`; it accepts `[10, 11, 12, 0]`. Therefore the synthetic fourth row may write cache location zero and overwrite real KV state. Padding sequence-length metadata to four rows does not make this side effect safe.

The candidate also has no end-to-end DSA execution evidence. Its own focused tests validate helper output and constructed metadata shapes, not attention, indexer/top-k, DeepGEMM schedule use, and KV writes under one declared extent.

Environment: AMD Instinct MI350X (`gfx950`), ROCm 7.2.26015, torch 2.11.0+rocm7.2. Python imports resolved to `/job/repo/python/sglang`. The patch changes no native files; `repository-environment.json` records no native build target, so no rebuild was applicable. CUDA-only DeepGEMM behavior and distributed attention TP are unverified. No suitable DeepSeek DSA weights/fixture were available; the provided tiny Llama fixture cannot qualify this architecture.

Raw commands and outputs are retained in `raw/`.
