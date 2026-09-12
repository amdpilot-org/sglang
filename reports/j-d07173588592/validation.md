# Validation notes

The original implementation loaded `ComponentLoadSpec` entries in a single
Python loop. Before the change, the new overlap regression failed with
`threading.BrokenBarrierError`, and the global-dtype adversarial test observed
both loader contexts active simultaneously. After the change, the complete
focused suite passed.

Raw evidence:

- `raw/focused-tests-after.txt`: focused compatibility and regression suite.
- `raw/gpu-numerical.txt`: one-GPU execution against an independent NumPy
  reference.
- `raw/host-memory-transfer.txt`: pageable versus pinned host-transfer probe.

The full Qwen-Image checkpoint was not present and was not downloaded. Therefore
this result is recorded as `candidate_verified`, not `fixed`: it validates the
loading contract and GPU execution, but not the real-model launch-time target.

The issue's wake/refit and pinned-memory observations remain follow-up work.
Current sleep/wake restoration is implemented separately in
`memory_occupation_controller.py`, and relevant safetensor readers explicitly
map checkpoints on CPU. Neither path is presented as solved by this change.
