# Independent review of PR 2514

Upstream issue: https://github.com/sgl-project/sglang/issues/29024

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2444

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2538

Candidate: https://github.com/amdpilot-org/sglang/pull/2514 at `a3d0a65f1042501891f0a067779cf28cf9e118d5`

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue on the recorded base. It fills the missing wrapper `rids` and `http_worker_ipcs` for tokenized generate and embedding batches while retaining item-level `http_worker_ipc` routing.

The failure was independently reproduced at base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`: both tokenized batch types retained `rids=None` and `http_worker_ipcs=None`. The same probe passed at the exact candidate commit, including empty batches and caller-supplied wrapper IDs. The candidate regression suite passed, and an independent source-checkout server run completed native batch generation with two tokenizer workers and tokenizer batch encoding enabled.

The issue text's separate scalar `http_worker_ipc` concern does not apply to this prepared base. `_handle_output_by_index` returns a one-item `BaseBatchReq`, `MultiDetokenizerRouter` assigns its one-element `http_worker_ipcs`, and `MultiHttpWorkerDetokenizerMixin` handles it through the batch branch. Adding a scalar field is therefore unnecessary here.

## Scope and limitations

- Imports resolved to `/job/repo/python/sglang/...`, so tests exercised the checked-out source rather than another installed SGLang copy.
- No C++/FlyDSL/native source changed. A native rebuild was not applicable. The imported AITER extension was `/tmp/amdpilot-repo-j-757bd0388cea/cache/aiter/module_aiter_core.so`.
- GPU execution used one AMD Instinct MI355X (`gfx950`) with ROCm 7.2 and Torch 2.11.0.
- The deterministic two-layer random Llama fixture validates transport, routing, batch tokenization, and engine execution. It does not establish model quality, other architectures, tensor/data parallelism, multi-GPU, or multi-node behavior.
- Shutdown tracebacks in the retained server log occur after the owned server process group receives SIGTERM; the probe completed first and the runner returned zero.

Raw outputs and the reusable independent probe are in `evidence/`.
