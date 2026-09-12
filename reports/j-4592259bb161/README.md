# Independent review of PR 665

Reviewed candidate: https://github.com/amdpilot-org/sglang/pull/665

Exact reviewed commit: `d6959386c233a1fcb0d27030bc9f576ddd7ca26e`

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The candidate does not contain a sampling-precedence implementation change. It is one
commit directly on the prepared base and adds only a prior review report, a deterministic
tiny-model generator, and an HTTP probe. Its report describes testing the different commit
`fc25fb0ed902605ed7aaf5592a72e42979219965` from PR 529. Evidence about that separate
revision does not verify PR 665 itself.

The original request-path reproduction fails identically on the base and exact candidate.
For an omitted chat request, `ChatCompletionRequest.to_sampling_params()` eagerly emits
`temperature=1.0`, `top_p=1.0`, `top_k=-1`, and `presence_penalty=0.0`; those values overwrite
server preferences `0.7`, `0.8`, `20`, and `1.5` in the tokenizer-manager merge.

The candidate's generated fixture was reproducible: `model.safetensors` had SHA256
`1407483bbc21bd0a4f84eb72acb2eb0d4b23019e5db843029e151f348e8fe108`, matching its report.
I then started the exact candidate on one AMD Instinct MI355X (`gfx950`) with
`torch_native` attention, prefill/decode graphs disabled, and an explicitly bounded 4096-token
pool. Its own probe exited 1 under `--expect-chat-precedence`. All eight requests returned
HTTP 200, both streams ended in `[DONE]`, and the two-prompt completion returned two choices,
but omitted chat output (` two nine C alpha one beta`) differed from explicit-preferred output
(` A C assistant gamma six F F`). Thus the protocol smoke passes while the original contract
still fails.

The first server attempt used the default memory sizing and failed after allocating a
262.5-GB KV cache; bounding `--max-total-tokens 4096 --mem-fraction-static 0.1` resolved that
fixture-specific startup issue. A non-fatal Torch Dynamo metrics serialization error appeared
during warmup. No C++ or other native source changed, so no native rebuild was applicable.
The imported protocol source was `/job/repo/python/sglang/srt/entrypoints/openai/protocol.py`.

Raw request-path and live-probe evidence was retained outside the revision-switching checkout
at `/job/review-evidence-j-4592259bb161/`.
