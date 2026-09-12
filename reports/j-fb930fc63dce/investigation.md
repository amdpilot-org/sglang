# Independent review of candidate 26cf569fbb512f4b6eca6d8fdba11983362874ee

Upstream issue: https://github.com/sgl-project/sglang/issues/31084

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2308

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2366

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2332

## Verdict

Recommendation: **accept**. The candidate fully resolves the original issue by
implementing the explicitly permitted fail-fast guard. It does not implement
cross-worker synchronization; instead, all three dynamic LoRA control entry
points return a clear error whenever `tokenizer_worker_num > 1`, before backend
communication or tokenizer-local registry mutation. Startup-preloaded adapters
and single-worker dynamic operations remain allowed.

## Base reproduction

The image-prepared checkout was exactly the requested base commit
`358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no difference to record.
I temporarily placed the candidate's regression test on that base while leaving
the production source unchanged. It produced four issue-specific failures and
two passes:

- both dynamic load variants incorrectly returned success with two tokenizer
  workers;
- unload returned the misleading worker-local `does not exist` error rather
  than a multi-worker incompatibility error;
- the single-worker and startup-preload boundaries passed.

Raw output: `raw/base-candidate-regression.log`.

## Exact candidate verification

I detached the prepared repository at exact commit
`26cf569fbb512f4b6eca6d8fdba11983362874ee`. With
`PYTHONPATH=/job/repo/python`, both `sglang` and
`tokenizer_control_mixin.py` imported from `/job/repo/python`, confirming that
the checked-out source was exercised rather than another installed copy.

The candidate changes only Python source/tests/reports. It has no C++, HIP,
FlyDSL, extension, or other native-code change, so no native rebuild applies.

The candidate regression plus related tokenizer-manager tests passed 16/16.
An independent adversarial test passed 3/3 and covered:

- path load, tensor load, and unload with three tokenizer workers;
- rejection before any backend communicator call or registry mutation;
- unchanged single-worker tensor loading and registration; and
- the more specific `LoRA is not enabled` error when LoRA itself is disabled.

Source inspection found no separate dynamic LoRA `from_distributed` endpoint in
this revision. The distributed-update symbols are weight-update APIs, while the
dynamic tensor form reaches `load_lora_adapter_from_tensors`; that handler is
guarded by the candidate.

## Related work inspected

Upstream PR https://github.com/sgl-project/sglang/pull/31724 remains open and
uses the same guard direction and files. Upstream PR
https://github.com/sgl-project/sglang/pull/36487 is closed. Neither is present
in the recorded base.

## Environment and limitations

The prepared environment uses Torch `2.11.0+rocm7.2` with HIP `7.2.26015` and
exposes one AMD Instinct MI350X. This is not a gfx950 system. No GPU computation
was performed because the reviewed change is a tokenizer-process control-plane
guard with no numerical, compiler, kernel, or model-architecture claim.

No model weights or full HTTP server were used. Consequently, real-model
serving transport, semantic accuracy, and multi-node/distributed execution were
not tested. Those limitations do not leave a counterexample to the selected
guard contract: the actual handlers and registries were exercised directly,
and every dynamic LoRA mutation entry point in this source revision is guarded.
