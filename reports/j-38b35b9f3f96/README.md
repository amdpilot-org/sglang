# DeepSeek-V4 decode retraction correction generation 2

Upstream issue: https://github.com/sgl-project/sglang/issues/33385

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2246

Parent candidate: https://github.com/amdpilot-org/sglang/pull/2155 at `19e9c3719ed7600c637fafbe14890b87113666a3`

Parent independent review: https://github.com/amdpilot-org/sglang/pull/2211

The review's concrete source counterexample was reproduced unchanged against
the exact candidate: the normal decode enqueue called
`add(req, is_retracted=False, is_rebootstrap=False)`, breaking the established
call contract tested by `test_decode_mode_assigns_default_priority_before_prealloc_queue`.

The correction preserves the candidate's valid DeepSeek-V4 handling: disabled
CPU-tensor offload selects PD rebootstrap, `NotImplementedError` from an
unsupported KV pool selects rebootstrap, host-pool exhaustion remains an abort,
and unrelated copy errors propagate. It now forwards `is_rebootstrap` only for
the true case, retaining the prior normal/retracted call shape. A boundary test
asserts that the true rebootstrap marker still reaches the preallocation queue.

The exact candidate test failed before; the corrected focused set passed 48
tests and 5 subtests. The supported MHA exact-value GPU backup/restore test also
passed on one AMD Instinct MI350X (gfx950).

The reported tp8/dp8 DeepSeek-V4 MTP4 HiSparse KV-full serving race was not run.
The model weights and distributed architecture were unavailable, so runtime
behavior, semantic output, and race timing remain unverified. The single-GPU MHA
test does not qualify DeepSeek-V4 or the distributed workload. No native source
changed and no native rebuild was applicable.
