# Independent review of amdpilot-org/sglang PR 3463

Candidate reviewed exactly at `2cf88b31bfc935a8f47b2f33332950c3a3b57be4` against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Recommendation: **accept**. The candidate fully resolves the original issue within the tested contract.

The prepared base reproduced the original defect: baked per-layer scales had correct non-unit runtime values, but `load_kv_cache_scales` still emitted the blanket fallback warning. The candidate records checkpoint calibration provenance before the sentinel is replaced, then diagnoses the actual post-load state. Fully calibrated baked layers no longer warn, absent calibration retains the fallback warning, and partial calibration reports the missing-layer count. A legitimate exact `1.0` scale and the supported single-tensor scale duplication path remain distinguishable from missing calibration.

The candidate's focused regression passed, as did the existing compressed-tensors KV-cache method-selection tests. An independent adversarial check used actual `RadixAttention` and `BaseKVCacheMethod` objects on an AMD Instinct MI350X and verified baked, single-tensor, missing, partial, and external runtime scale values. Imports were confirmed to resolve to `/job/repo/python/sglang`, not an installed SGLang wheel.

No native source changed, and the prepared environment reports no native build target, so no rebuild was applicable. No GB10 run, full model-weight serving run, semantic-accuracy qualification, or distributed workload is claimed.

Upstream issue: https://github.com/sgl-project/sglang/issues/31224

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/3443

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/3466
