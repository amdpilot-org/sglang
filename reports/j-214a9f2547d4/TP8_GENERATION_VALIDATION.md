# DeepSeek-V4 Flash TP8 generation validation

## Status

**Generation acceptance: FAILED.**

Official DeepSeek-V4 chat framing did not fix the prior longer-generation failure. With `temperature=0`, `top_p=1`, `seed=12345`, `max_tokens=256`, and `ignore_eos=false`, all four required cases failed coherence, completion, repeat-consistency, or code-correctness checks in both `/v1/chat/completions` and officially framed raw `/generate` mode. No upstream 200-question result is claimed by this small sample, and this hardware is MI300X, not MI308X.

No production source change was made. The existing FP4-to-FP8 converter fix is preserved unchanged in the exact control commit.

## Source control

- Issue: `sgl-project/sglang#35122`
- Delivery branch: `amdpilot/j-214a9f2547d4`, cut from `amdpilot-org/sglang` `main`
- Original PR57 source base: `db272201a2dbd72e5699e443240a851f1313ad45`
- Exact control commit: `484c2286c993d36e862343c390a77439a003d244`
- Prior TP8 source commit: `082ad8ce15176ac80fa0afd4daa3a3ef71bbb126`
- PR56 source commit: `710dc165936d617826c492016ed9189875376bdc`
- The control worktree remained clean at the exact commit before startup and after all requests.
- `python/sglang/srt/layers/quantization/fp8.py` and `test/registered/unit/layers/quantization/test_fp8_rocm_fp4_dequant.py` are byte-identical between the prior TP8 source and the exact control commit.

The delivery tree contains only this report and the diagnostic scripts. The server source was never taken from the delivery tree or current `main`.

## Checkpoint identity

- Read-only path: `/models/DeepSeek-V4-Flash-0731`
- Identity: `deepseek-ai/DeepSeek-V4-Flash-0731`, revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`
- Index SHA256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`
- Index entries: 72,317
- Safetensor shards: 48
- Safetensor bytes: 166,886,535,336
- No weights were downloaded, repacked, or modified.

## Runtime

- Container image recorded by the job: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- Python: 3.10.12
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- Transformers: 5.12.1
- Triton: 3.7.0
- AITER: `/sgl-workspace/aiter/aiter/__init__.py`
- GPUs: eight AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 304 CUs each
- Attention backend: `dsv4`
- FlashMLA backend: `triton`

Imported source paths were asserted under `/tmp/sglang-pr57-control/python` for:

- `sglang`
- `sglang.srt.layers.quantization.fp8`
- `sglang.srt.layers.attention.hip_flash_mla`
- `sglang.srt.layers.attention.deepseek_v4_backend_hip_radix`
- `sglang.kernels.ops.attention.nsa_triton_decode`

## Official framing and EOS

The checkpoint has no ordinary HuggingFace Jinja chat template. The exact control’s `encoding_dsv4.py` and `serving_chat.py` use the official DeepSeek-V4 encoder.

- BOS token ID: 0
- EOS token ID: 1
- EOS text: `<｜end▁of▁sentence｜>`
- Official non-thinking mode: `thinking_mode="chat"`
- Server `reasoning_parser`: `None`
- Effective chat mode: `CHAT`
- Assistant generation prefix: `</think>`

The arithmetic prompt was encoded as:

```text
<｜begin▁of▁sentence｜><｜User｜>Explain 12 + 34 in two sentences.<｜Assistant｜></think>
```

Token IDs:

```text
[0, 128803, 65106, 223, 736, 940, 223, 2012, 295, 1234, 15174, 16, 128804, 128822]
```

All raw `/generate` requests used the exact same official formatted prompt and `stop_token_ids=[1]`. Every request used `ignore_eos=false`.

## Server launch

Wrapper: `reports/j-214a9f2547d4/scripts/run_tp8_server.sh`

Explicit environment:

```text
SGLANG_USE_AITER=1
SGLANG_DSV4_FP4_DEQUANT=1
SGLANG_HACK_FLASHMLA_BACKEND=triton
PYTHONPATH=/tmp/sglang-pr57-control/python
```

Server flags:

```text
--host 127.0.0.1
--port 31322
--model-path /models/DeepSeek-V4-Flash-0731
--tp 8
--cuda-graph-max-bs-decode 8
```

All eight ranks logged:

```text
Dequantized FP4 expert weights to FP8.
```

All eight ranks completed target decode CUDA graph capture for batch sizes `[1, 2, 4, 8]`. The server became healthy after 660 seconds.

## Acceptance cases

Each fixed case was run twice in official chat mode and twice in officially framed raw `/generate` mode. All requests returned HTTP 200.

| Mode | Case | Repeat 1 | Repeat 2 | Repeats equal | Accepted |
|---|---|---|---|---|---|
| chat | arithmetic | length, 256 tokens | length, 256 tokens | no | no |
| chat | day_night | stop, 178 tokens, EOS | length, 256 tokens | no | no |
| chat | france | length, 256 tokens | length, 256 tokens | no | no |
| chat | sum_squares | length, 256 tokens | length, 256 tokens | no | no |
| raw | arithmetic | stop, 38 tokens, EOS | length, 256 tokens | no | no |
| raw | day_night | length, 256 tokens | length, 256 tokens | no | no |
| raw | france | length, 256 tokens | length, 256 tokens | no | no |
| raw | sum_squares | length, 256 tokens | length, 256 tokens | no | no |

`finish_reason=length` is incomplete and is not accepted as a coherence pass.

### Representative failures

Chat arithmetic repeat 1 began:

```text
12 plus 34 is 46, with a full stop, 1 2 (3) 0. 0. 1 2 4 6 0. 1 2 4 6 0.
```

Chat arithmetic repeat 2 began:

```text
12 + 34 = 46, because 12 is added to 34, then 34 is added to 12, so 12 (di) 46.
```

Chat day/night repeat 1 mentioned Earth’s rotation but repeated malformed sentences until EOS. Repeat 2 omitted Earth and looped until the length cap.

Chat France repeat 1 began:

```text
The capital of France is the city of Paris, which is a major financial and economic center.
```

It then degenerated into repeated fragments such as `A city. A city. A city.`

Raw France repeat 2 began:

```text
The capital of the capital is Paris, and the capital of the capital of the capital...
```

The `sum_squares` outputs either had no code block or an incomplete code block with no return statement or expected values. The generated code was inspected before bounded execution. Because no complete Python code block was present, it was not executed.

## Weight and attention diagnosis

Representative routed expert tensors were inspected directly from the original safetensors shards:

- `layers.0.ffn.experts.0.w1.weight`: `I8`, shape `(2048, 2048)`
- `layers.0.ffn.experts.0.w1.scale`: `F8_E8M0`, shape `(2048, 128)`
- `layers.0.ffn.experts.0.w2.weight`: `I8`, shape `(4096, 1024)`
- `layers.0.ffn.experts.0.w2.scale`: `F8_E8M0`, shape `(4096, 64)`
- `layers.0.ffn.experts.0.w3.weight`: `I8`, shape `(2048, 2048)`
- `layers.0.ffn.experts.0.w3.scale`: `F8_E8M0`, shape `(2048, 128)`

The same packed-I8 plus `F8_E8M0` layout was present for experts 0, 1, and 255 in layers 0, 21, and 42.

The live AITER fused-MoE path logged:

```text
torch.float8_e4m3fnuz
QuantType.per_1x128
```

The active attention path was:

```text
attention_backend=dsv4
SGLANG_HACK_FLASHMLA_BACKEND=triton
triton_fp8_attention_fwd
```

The exact control’s `cast_e2m1fn_to_e4m3fn` unpacks packed FP4 nibbles, expands the 32-element FP4 blocks into 128-element FP8 blocks, and emits `float8_e4m3fn` weights with `float8_e8m0fnu` block scales. The converter source is unchanged from the previously validated PR56/PR57 source.

Because official framing still produced incoherent and inconsistent output, the remaining likely scope is the DSV4 decoder path, routed-expert dequantization/layout, or active Triton attention path. No evidence justified a blind kernel change.

## Reproduction

```bash
git worktree add --detach /tmp/sglang-pr57-control \
  484c2286c993d36e862343c390a77439a003d244

export PYTHONPATH=/tmp/sglang-pr57-control/python
python reports/j-214a9f2547d4/scripts/inspect_dsv4_encoding.py

reports/j-214a9f2547d4/scripts/run_tp8_server.sh &
python reports/j-214a9f2547d4/scripts/validate_dsv4_generation.py
python reports/j-214a9f2547d4/scripts/evaluate_dsv4_results.py
```

The wrapper in this PR hard-asserts the exact control HEAD, clean `python/sglang`, and both imported module paths before launching the server.

## Artifacts

Full raw request and response JSON, token IDs, logs, and source checkpoints are preserved under `/job`:

- `/job/raw/dsv4_official_encoding.json`
- `/job/raw/dsv4_generation_validation.json`
- `/job/raw/dsv4_acceptance_summary.json`
- `/job/logs/server.log`
- `/job/logs/generation_validation.log`
- `/job/logs/acceptance_evaluation.log`
- `/job/logs/runtime_identity.json`
- `/job/logs/prelaunch_control_assertions_retry1.txt`
- `/job/logs/postrequest_control_assertions.txt`
- `/job/logs/routed_weight_layout.txt`
- `/job/checkpoints/initial/`
- `/job/recovery-latest.patch`

The server received a targeted SIGTERM to its exact PID, logged a graceful drain with zero remaining requests, and released port 31322. All eight GPUs were idle afterward.

## Left undone

- No production fix was attempted because no evidence isolated a specific incorrect kernel or tensor transform.
- The unchanged converter regression tests were not repeated; they had already passed in the prior controlled job.
- No 200-question benchmark was run.
- No claim is made that issue 35122 is resolved.
