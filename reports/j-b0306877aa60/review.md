# Independent review of PR 1740 at `9af7b07`

Recommendation: **accept**. The exact candidate fully resolves the original issue as stated.

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, an independent harness created four allocated slots with `_active_workers == [0, 1]`, set `workers[2:]` to `None`, and routed `bootstrap_room=2`. The implementation computed `2 % len(workers)`, selected slot 2, and raised `AttributeError` in `sock_send`.

At candidate commit `9af7b07e1fd3d27c76f1ea374bff8d39b746d734`, the scheduler computes modulo over `_active_workers` and then maps the logical result back to the physical slot. This sends room 2 to slot 0 for the reported topology. `_active_workers` is the controller's maintained ordered active topology and is already used by round-robin and direct-rank validation.

The candidate is a source fix with regression coverage, not test-only hardening. Its six focused tests and full 20-test controller module pass. Independent adversarial checks also pass for sparse active slots `[1, 4]`, absent sockets in inactive slots, large room IDs, repeated modulo mapping, and an empty active domain. No remaining counterexample to the original contract was found.

The supplied interpreter imported `sglang` and `data_parallel_controller.py` from `/job/repo/python`, confirming the candidate checkout was tested rather than an installed unrelated copy. The candidate changes no C++/HIP/native source, so a native rebuild was not applicable.

Environment: Linux x86_64, Python 3.12.3, Torch 2.11.0+rocm7.2, HIP 7.2.26015, with one AMD Instinct MI350X (`gfx950:sramecc+:xnack-`) visible. GPU execution was intentionally not claimed: this failure occurs in deterministic Python routing before worker/model execution. No full serving deployment, model weights, semantic accuracy check, or multi-node workload was exercised.

Raw command output, candidate diff, import paths, issue/PR metadata, and environment details are retained in `evidence/`.
