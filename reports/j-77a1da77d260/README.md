# Live qualification of PR 529

Candidate PR: https://github.com/amdpilot-org/sglang/pull/529

Pinned candidate head: `fc25fb0ed902605ed7aaf5592a72e42979219965`

Prepared baseline: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The server command used `--attention-backend torch_native --disable-cuda-graph`
on one AMD Instinct MI350X (`gfx950`). Both graph phases resolved to `disabled`.
The preferred internal parameters were:

```json
{"temperature":0.0,"top_p":0.55,"top_k":1,"presence_penalty":1.25,"repetition_penalty":1.1,"sampling_seed":17}
```

Generate the immutable local fixture with:

```bash
/tmp/amdpilot-repo-j-77a1da77d260/venv/bin/python \
  scripts/ci/generate_tiny_sampling_model.py \
  /tmp/amdpilot-repo-j-77a1da77d260/models/tiny-llama
```

After starting the server, probe it with:

```bash
/tmp/amdpilot-repo-j-77a1da77d260/venv/bin/python \
  scripts/ci/probe_sampling_precedence.py \
  --base-url http://127.0.0.1:PORT --expect-chat-precedence
```

The assertion fails on the baseline and passes at the exact candidate head. Baseline
omitted chat output was ` five ten nine H user`, while explicit preferred output was
` A C assistant gamma six F F`. At the candidate, both were
` A C assistant gamma six F F`. All requests returned HTTP 200, both streaming APIs
ended with `data: [DONE]`, and the batched completion returned two indexed choices.

The generated model's `model.safetensors` SHA256 is
`1407483bbc21bd0a4f84eb72acb2eb0d4b23019e5db843029e151f348e8fe108`.
Complete raw responses and the candidate server log are retained under
`/tmp/amdpilot-repo-j-77a1da77d260/artifacts/raw`.
