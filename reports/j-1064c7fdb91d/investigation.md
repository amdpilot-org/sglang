# Investigation: Kimi-K3 cross-prompt reasoning leakage

Upstream issue: https://github.com/sgl-project/sglang/issues/34259

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1621

## Finding

The prepared main source already contains a concrete cross-request isolation fix that matches the mechanism in related issue https://github.com/sgl-project/sglang/issues/32089. Upstream PR https://github.com/sgl-project/sglang/pull/30177 merged on 2026-08-02, after the affected 0.5.17 report, and includes commit `208d2f9a0a61ed4b056523e740338218e4c26188` (`Fix hidden-state batching and graph reuse`).

The current implementation in `python/sglang/srt/managers/scheduler_components/batch_result_processor.py` advances `hidden_state_offset` by `extend_input_len` before deciding whether a request stores hidden states. Therefore a request with `return_hidden_states=False` still consumes its packed rows. The scheduler also snapshots `extend_input_len_per_req` when either logprobs or hidden states are requested, avoiding live overlap-scheduler mutation and avoiding the incorrect use of original prompt length for cached/chunked prefills.

No production source correction was justified on this base. This PR adds explicit regression coverage for the original mixed-request example and independent cached-prefix/zero-row boundaries.

## Reproduction evidence

With the old skip-without-advance behavior temporarily restored, the direct regression failed because the final offset was `3` instead of `8`. On the unmodified prepared source, the full test file passed (`9 passed`, plus two subtests), and the focused offset class passed (`3 passed`, plus two subtests).

On the assigned AMD Instinct MI350X (`gfx950:sramecc+:xnack-`), the production helper consumed an eight-row device tensor representing five non-requesting rows followed by three requesting rows. It returned offset `8` and copied `[[20.0], [21.0], [22.0]]`, matching the independent expected values rather than the first request's rows.

Raw command output was retained during the investigation under `/tmp/amdpilot-repo-j-1064c7fdb91d/evidence/`. The tested source is `/job/repo`; the interpreter is `/tmp/amdpilot-repo-j-1064c7fdb91d/venv/bin/python`.

## Limitations

Kimi-K3 weights were unavailable and the report is random with no deterministic reproducer. The original environment used eight NVIDIA B300 GPUs, while this job assigned one AMD gfx950 GPU. Consequently this work does not claim a full Kimi-K3, B300, tensor-parallel, multi-node, semantic-accuracy, or production HTTP reproduction. It verifies the specific cross-request packed-hidden-state mechanism and records that the reporter also stated the symptom was no longer observed on main.
