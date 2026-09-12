# Independent review of PR 2413

Upstream issue: https://github.com/sgl-project/sglang/issues/32572

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2447

Candidate: https://github.com/amdpilot-org/sglang/pull/2413 at exact commit `b6def9877421da520d9cfbc77ab82e5afdfb9ac3`.

## Recommendation

Accept. The candidate fully resolves the source-level contract demonstrated by
the original issue and the concrete MOSS-VL counterexample from the earlier
independent review. This is a functional fix, not merely test hardening: it
adds `linear_fc1`/`linear_fc2` normalization and dimensions, routes Qwen3-VL
deep-stack modules and adapter weights by their indexed merger paths, and gives
the unindexed MOSS-VL merger a model-specific logical layer.

## Failing-before evidence

The checkout began on the recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, with no difference from the
image-prepared revision. Running the candidate's regression files against that
base produced 12 failures and 4 passes. The failures included the reported
`linear_fc1`/`linear_fc2` dimension lookup, all indexed Qwen3-VL deep-stack
layer mappings, both unindexed MOSS-VL merger registrations, and both MOSS-VL
adapter-weight placements. See `raw/base_candidate_regressions.txt`.

The interpreter imported `sglang`, `lora_manager.py`, and `utils.py` from the
prepared `/job/repo/python` tree, not an installed wheel. See
`raw/base_import_gpu.txt`.

## Exact-candidate evidence

After a detached checkout of exact commit `b6def9877421da520d9cfbc77ab82e5afdfb9ac3`,
the same prepared interpreter again resolved imports to `/job/repo/python`.
The candidate's two focused test files passed all 16 tests.

An independent nine-case suite additionally exercised:

- real `LoRAManager.init_lora_modules` traversal over three Qwen3-VL
  `deepstack_merger_list` entries and verified each pair lands in its indexed
  logical layer;
- Qwen3-VL adapter-weight placement for the captured `0.linear_fc1` and
  `2.linear_fc2` naming form;
- rejection of four near-miss MOSS-VL paths;
- precedence of ordinary decoder `layers.N` resolution over the model hook;
- the intentional absence of routing for Qwen3-VL's unindexed primary merger;
- MOSS-VL merger dimensions derived from a varied deep-stack count rather than
  hard-coded production dimensions.

All nine passed. Python compilation of the changed LoRA/model modules also
passed. Raw outputs are retained under `raw/` and the independent test source
is `independent_contract.py`.

The unindexed Qwen3-VL primary `visual.merger` is not assigned a LoRA layer.
This is not a remaining counterexample to the reported adapter: the captured
EditScore adapter configuration targets only indexed `0`, `1`, and `2`
`linear_fc1`/`linear_fc2` modules, corresponding to Qwen3-VL's deep-stack
mergers. MOSS-VL has a genuinely unindexed merger and is explicitly handled.

## Environment and scope

The prepared environment has one AMD Instinct MI355X with
`gfx950:sramecc+:xnack-`, PyTorch `2.11.0+rocm7.2`, and ROCm 7.2.26015. The
original report used an NVIDIA H100 NVL, CUDA 13.2, Qwen3-VL-8B weights, and the
EditScore adapter. Those model weights were not prepared, so no full HTTP
server/model-load reproduction was performed and `gpu_execution` is reported
as false for the issue-specific review. The deterministic tests cover the
initialization, module registration, dimensions, and adapter routing that
caused the original startup exception, but not end-to-end generation quality.

No native source changed between the base and candidate, so no native rebuild
was required or performed. The candidate does contain trailing whitespace in
historical raw report artifacts; this does not affect runtime behavior.

