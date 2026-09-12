# Independent review of amdpilot-org/sglang PR 1051

Reviewed exact candidate commit `0ba6f8161b30353024ef292d56d29c5385b6b81d`
against recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` and the
original issue contract.

The base reproduces the issue-specific source defect: `AudioEncoderAttention`
does not pass `use_dp_attention_reduce`, although audio projection/QKV weights
are sharded by `attn_tp_size` and `attn_tp_rank`. The focused base regression
therefore observed a missing routing argument and exited 1.

On the exact candidate, the candidate regression passed both DP-attention
boundaries (2 passed). Independent adversarial checks additionally established:

- `False` retains the full-TP reduction selection and `True` selects the
  attention-TP group.
- Synthetic rank 0/rank 1 audio projection weights are distinct attention-TP
  shards and concatenate back to the original tensor.
- A direct `RowParallelLinear.forward` branch probe returned the sentinel from
  `attn_tp_group.all_reduce` when enabled and the full-TP all-reduce sentinel
  when disabled.
- `mimo_v2_asr.py` consumes the same `AudioEncoderMixin`, so its inherited audio
  encoder receives the same correction.

The imported source was `/job/repo/python/sglang/srt/models/mimo_audio.py`, not
an installed wheel. This is a Python-only change; no native source changed and
no native rebuild was applicable. Compileall and diff whitespace checks passed.

Recommendation: accept. The source change fully resolves the communicator
selection defect described by the issue. This conclusion is based on the actual
weight-sharding and reduction paths, not merely the candidate's prose or a GPU
smoke test.

The reported end-to-end failure could not be executed here: the assigned host
has one AMD Instinct MI355X (`gfx950`, ROCm 7.2), not 8 NVIDIA H200 GPUs, and the
MiMo-V2.5 weights were unavailable. Consequently the CUDA/NCCL watchdog hang,
TP=8/DP=2 serving request, FP8 model behavior, and post-request server recovery
remain topology/model-level verification limitations. No contrary source-level
counterexample was found.

Upstream issue: https://github.com/sgl-project/sglang/issues/37059

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1090
