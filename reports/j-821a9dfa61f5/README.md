# Developer reload implementation report

Outcome: **candidate_verified**

This change adds an opt-in, admin-authenticated `/dev/reload` API. It drains
inference through the existing model-update writer lock, broadcasts the same
module list to scheduler workers, reloads loaded pure-Python `sglang.*` modules,
updates existing class instances and direct-import aliases, recaptures target
and draft decode graphs, and resumes serving without loading weights.

The ROCm TP=1 integration used the campaign's deterministic two-layer Llama
fixture at commit `f1d603677ca76a9ea21124a544e405c5b0cbd315`. Greedy output IDs before
and after reload were `[12, 100, 124, 43, 124, 43, 124, 51]`; an independent
Hugging Face CPU generation produced the same IDs. The live server recaptured
decode graphs for batch sizes 1 and 2 in 0.48 seconds. Authentication and module
scope rejection were also exercised.

## Reproduce

Use the prepared interpreter and create the fixture outside the checkout, then
launch the server with `--enable-dev-reload --admin-api-key dev-secret` and the
tiny-model settings recorded in `result.json`. Send:

```bash
curl -X POST http://127.0.0.1:38121/dev/reload \
  -H 'Authorization: Bearer dev-secret' \
  -H 'Content-Type: application/json' \
  -d '{"modules":["sglang.srt.layers.activation"],"recapture_cuda_graph":true}'
```

Run the focused regression suite with:

```bash
/tmp/amdpilot-repo-j-821a9dfa61f5/venv/bin/python -m pytest -q test/srt/test_dev_reload.py
```

## Limitations

Only single-node TP=1 was available for GPU validation. Multi-rank fan-out uses
the existing scheduler broadcast/aggregation path but was not hardware-tested.
Native extensions, newly imported modules, changed class/object layout, process
topology, torch.compile artifacts, and JIT kernel changes still require restart.
Reload is not transactional if executing changed module code raises. The
fault-tolerant per-request scheduler loop proposed as the issue's other possible
direction is not implemented. Large-model timing and architecture-specific
semantic correctness were not tested by the synthetic Llama fixture.
