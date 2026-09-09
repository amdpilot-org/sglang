# TP8 FP8 software-conversion A/B result

## Verdict

**Whole-model semantic A/B: FAILED.** The isolated PR60 `fp8_utils.cuh` software-conversion patch did **not** turn the rigorously failed long-generation control into coherent, correct, repeatable, EOS-completed generation.

Both arms ran the same four fixed cases twice through `/v1/chat/completions` and twice through officially framed raw `/generate`. Every one of the eight mode/case groups failed acceptance in both arms. Startup, weight loading, and decode graph capture are not counted as semantic success.

## Source and patch isolation

- Control: detached `amdpilot-org/sglang` at `484c2286c993d36e862343c390a77439a003d244`, clean before and after the arm.
- Candidate: separate detached worktree at the same commit plus only `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/fp8_utils.cuh`.
- Baseline header SHA256: `6b686494e4ee7ac6f12f972e1033837ff020d1629e833e06f57f304ed0c5f022`.
- Patched header SHA256: `16ee2f59330081e0015390d4f895cbd7aa093b55ace86723a668699b130ff375`.
- The literal patch was checked with `git apply --check` before applying it only to the candidate.
- No merge, rebase, or cherry-pick was performed. PR60, PR61, and this report branch were never used as server/encoder source.

## Runtime and checkpoint

- Image requested for this job: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, expected image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Hardware: exactly eight AMD Instinct MI300X `gfx942` GPUs.
- Model mount: read-only `/models/DeepSeek-V4-Flash-0731`.
- Model identity: `deepseek-ai/DeepSeek-V4-Flash-0731`, revision `7872f01b1d1fe23eabc4c98448bffcef5a386062`.
- Index SHA256: `98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`.
- Index entries: 72,317.
- Safetensors: 48 shards totaling 166,886,535,336 bytes.
- No weights were downloaded, repacked, or modified.

## Binary isolation

Each arm used separate initially empty private caches for SGLang JIT, Triton, TorchInductor, DeepGEMM, CUTE AOT, FlyDSL, ccache, Numba, and XDG cache state. Arms ran sequentially in fresh Python/server processes.

The live 512-dimensional `fused_norm_rope_v2` producer was rebuilt from the arm-specific header:

- Control page-2 producer: `/job/caches/control/sglang/jit/gfx942/sgl_kernel_jit_dpsk_v4_fused_norm_rope_v2_fp32_t_512_64_2_false_16_false/...so`, SHA256 `60ea2f7eb6dc503f9f21ab874f2fd990ea9e1219d4f306c861ba6bca536e8a36`, dependency digest `6b686494e4ee7ac6f12f972e1033837ff020d1629e833e06f57f304ed0c5f022`.
- Candidate page-2 producer: `/job/caches/candidate/sglang/jit/gfx942/sgl_kernel_jit_dpsk_v4_fused_norm_rope_v2_fp32_t_512_64_2_false_16_false/...so`, SHA256 `7fa5f99850066459e156f349e66d16a39e834eb5d46a576c4c0a477c3ae70f57`, dependency digest `16ee2f59330081e0015390d4f895cbd7aa093b55ace86723a668699b130ff375`.
- Control page-64 producer SHA256: `c44854aab5a743de3f82f65c0fd12558843733f2dbd49883fc4cf1b780e598fa`.
- Candidate page-64 producer SHA256: `597688177c297cae5de2f643b60788e3f1844341f969e00e927f4d4a98d16fed`.

The source-derived ordinary page-256 pools resolve to C4 page size 64 and C128 page size 2. The loaded native template identities `512_64_2` and `512_64_64` corroborate those resolved page sizes. The deprecated `/get_server_info` field `c128_page_size=16` is not the resolved ordinary pool page size used by these producer templates.

Full worker mappings, loaded private `.so` paths, SHA256 hashes, `build.ninja`, generated `cuda.cu`, dependency manifests, and compiler command lines are preserved under `/job/evidence/{control,candidate}/binary-evidence*.json` and the per-producer evidence files. Immutable image-installed AITER libraries shared across arms are separately recorded with paths and SHA256 hashes under `shared-library-provenance*.json`; they are not consumers of the changed SGLang header.

## Server configuration

Both arms used the same explicit wrapper configuration:

```text
SGLANG_USE_AITER=1
SGLANG_DSV4_FP4_DEQUANT=1
SGLANG_HACK_FLASHMLA_BACKEND=triton
PYTHONPATH=<arm>/python
```

Both used the same spaced server arguments:

```text
python -m sglang.launch_server
--host 127.0.0.1
--port 31322
--model-path /models/DeepSeek-V4-Flash-0731
--tp 8
--cuda-graph-max-bs-decode 8
--random-seed 12345
```

No FP8 macros were hand-added or removed. Both arms used the normal production `gfx942` architecture flags emitted by the JIT build system.

Imported `sglang.__file__`, `sglang.srt.layers.quantization.fp8.__file__`, and `encoding_dsv4.__file__` were asserted to resolve to each arm’s private source tree before launch. All eight ranks logged FP4-to-FP8 expert dequantization and completed target decode graph capture for batch sizes `[1, 2, 4, 8]`.

## Official framing and requests

The checkpoint has no ordinary HuggingFace Jinja chat template. Each arm’s exact `encoding_dsv4.py` was used in official non-thinking `thinking_mode="chat"` with the official reasoning-effort profile.

- BOS token ID: 0
- EOS token ID: 1
- EOS text: `<｜end▁of▁sentence｜>`
- Assistant generation prefix: `""`

Each fixed case was sent twice through `/v1/chat/completions` and twice through raw `/generate` using that arm’s exact formatted prompt and prompt token IDs. Sampling was temperature `0`, top-p `1`, seed `12345`, max tokens `256`, and `ignore_eos=false`. Raw requests used `sampling_seed=12345`, `max_new_tokens=256`, and `stop_token_ids=[1]`.

## Acceptance

| Arm | Mode | Case | Accepted | Repeats equal | All stop/EOS | Semantic checks |
|---|---|---|---:|---:|---:|---:|
| Control | chat | arithmetic | no | no | no | no |
| Control | chat | day_night | no | no | no | no |
| Control | chat | france | no | no | no | partial |
| Control | chat | sum_squares | no | no | no | no |
| Control | raw | arithmetic | no | no | no | no |
| Control | raw | day_night | no | no | no | no |
| Control | raw | france | no | no | no | no |
| Control | raw | sum_squares | no | no | no | no |
| Candidate | chat | arithmetic | no | no | no | no |
| Candidate | chat | day_night | no | no | no | no |
| Candidate | chat | france | no | no | no | no |
| Candidate | chat | sum_squares | no | no | no | no |
| Candidate | raw | arithmetic | no | no | no | no |
| Candidate | raw | day_night | no | no | no | no |
| Candidate | raw | france | no | no | no | no |
| Candidate | raw | sum_squares | no | no | no | no |

Finish-reason counts across each arm’s 16 requests:

- Control: 14 `length`, 2 `stop`.
- Candidate: 15 `length`, 1 `stop`.

No response pair met exact repeated-response consistency. No `sum_squares` response contained a complete executable function; bounded execution was therefore not able to produce the required `[]->0`, `[1,2,3]->14`, or `[-2,3]->13` result. The candidate showed occasional local improvements, such as an arithmetic response containing `46`, but those responses still repeated, became incoherent, hit the 256-token length cap, and failed repeat consistency. A candidate raw France response stopped with EOS, but its repeat did not, and the pair was not accepted.

No explicitly labeled 512+ secondary requests were added because the observed 256-token failures were repetitive/incoherent rather than legitimate answers that merely needed a higher cap.

## Limits

- This is a small declared sample, not a 200-question or upstream-score evaluation.
- The result does not prove issue 35122 is fixed.
- The already qualified PR60/PR61 single-GPU and cache comparisons remain unchanged and are not rerun here.
- Historical PR59/raw failures are preserved and were not overwritten.
- No speculative decoder/cache or graph-enablement discriminator was bundled into this A/B.

## Preservation

- Raw requests and complete responses: `/job/raw/control` and `/job/raw/candidate`.
- Per-request flushed JSONL and individual JSON records are present for all 32 requests.
- Token ID traces, full text, finish/matched-stop state, latency, and arm identity are included in every raw record.
- Acceptance summaries: `/job/evidence/control/acceptance-summary.json` and `/job/evidence/candidate/acceptance-summary.json`.
- Combined A/B summary: `/job/evidence/ab-acceptance-summary.json`.
- Server logs: `/job/logs/control/server.log` and `/job/logs/candidate/server.log`.
- Startup, shutdown, model, GPU, image, cache, and binary evidence are under `/job/checkpoints/initial`, `/job/evidence`, `/job/logs`, and `/job/caches`.
- Top-level recovery patch: `/job/recovery-latest.patch`.
- Draft PR: `https://github.com/amdpilot-org/sglang/pull/62`
