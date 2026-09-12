# Independent review of candidate PR 571

Upstream issue: https://github.com/sgl-project/sglang/issues/38817

Mirror issue: https://github.com/amdpilot-org/sglang/issues/575

Candidate reviewed: `e23d8abd599e9e43ab9fffdb91ad2043a4cf37bf`

Recommendation: **accept**. The candidate fully resolves the reproducible original-issue contract.

On prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`, both public norm implementations returned FP8 when given FP8 input. Concatenating that result with the BF16 MTP hidden state raised the exact reported error. Current NEXTN implementations contain this norm-to-concat pattern.

At the exact candidate commit, FP8 norm inputs are accumulated through the native FP32 path and returned in the norm weight/model activation dtype. The candidate regression passed, as did independent cases spanning all four Torch FP8 formats, both RMSNorm variants, irregular dimensions, residual and post-residual inputs, AITER on/off, and ordinary FP16/BF16/FP32 behavior. GPU results were compared with independently calculated FP64 references.

The original report does not specify a model/checkpoint or complete service command, so the exact 0.5.6 server launch was not available. The precise failing operation and current implementation path were nevertheless reproduced directly. Testing was limited to the assigned AMD Instinct MI350X (`gfx950`) with ROCm 7.2; CUDA/NVIDIA and other GPU architectures were not available.

No C++ or other native source changed in the candidate. No native rebuild was applicable. The imported implementation was `/job/repo/python/sglang/srt/layers/layernorm.py` from the checked-out source tree.

Raw evidence is retained in `reports/j-d56b9e23145e/raw/`.
