# Independent review of PR 3361

Candidate: `ea0cfb324846ce69627ceb74aa0488a9a0502d94`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

## Finding

The candidate is a partial startup guard, not a full fix for the original issue. It rejects the exact string-equal combination of GLM5, EAGLE3, and the bundled/same-path draft. It does not make the reported EAGLE3/MTP configuration work, validate the proposed NEXTN/EAGLE alternative with GLM-5.3-Flash, address the reported acceptance of approximately 1.0, or establish which hidden representation the bundled head was trained to consume.

The guard is also bypassable for the same local checkpoint expressed through an equivalent path. The independent adversarial test allowed `/models/glm5/`, `/models/./glm5`, and a different mount/symlink spelling when the target was `/models/glm5`. Once bypassed, the unchanged EAGLE3 capture path remains capable of producing the original incompatible width.

## Failing-before evidence

On the prepared base, the candidate regression file is absent, so invoking it fails collection. More importantly, an implementation-level reproduction called the actual GLM5 capture methods: the default layers were `[2, 22, 42]`; with `H=4096`, `hc_mult=4`, and `dflash_capture=False`, each captured state remained width 16384 and packing three captures produced width 49152, while `deepseek_nextn.py` expects the previous hidden input to match the H-wide side of `eh_proj`.

The prepared base already contracts mHC state for DFLASH capture only. That existing behavior does not contract EAGLE3 captures.

## Candidate evidence

- The candidate's five validator tests plus the pre-existing DFLASH capture test passed: 6 passed.
- The imported `sglang` and `speculative_hook` modules came from `/job/repo/python`, and inspection confirmed `_handle_eagle_family` calls the new validator.
- No native source changed, so a native rebuild was not applicable. The exercised code was Python source from the checkout.
- On the assigned AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), the pre-existing DFLASH-only contraction transformed `[3, 16384]` to `[3, 4096]` with maximum absolute difference 0.0 against an independent reshape-and-mean reference. With `dflash_capture=False`, the same actual helper retained `[3, 16384]`, confirming that this GPU result does not validate EAGLE3.

## Classification

Recommendation: `request_changes`.

This is partial defensive hardening. The exact tested spelling gets a clearer early error, but the original requested feature remains unsupported and the rejection itself is incomplete. The assertion that the bundled head is trained for the final target hidden state through NEXTN/EAGLE remains unverified without weights, authoritative checkpoint metadata/training evidence, or a full model acceptance-quality run.

## Environment limitations

The host has one AMD Instinct MI355X gfx950 GPU with ROCm 7.2 and Torch 2.11.0+rocm7.2. The reported 4x NVIDIA H20 TP=4 topology and GLM-5.3-Flash FP8 weights were unavailable. Therefore no full server reproduction, CUDA `fused_eh_norm` execution, model semantic validation, acceptance-length measurement, throughput benchmark, or multi-node claim is made.

Raw command output is retained in this report directory.
