# Independent review of amdpilot-org/sglang PR 560

Reviewed exact candidate commit `4920768354a34e14f43fb6d0c56ab491d608fa96` against prepared base `358c163250ad3b1f62939b01ce1314a0a31a0365`.

Upstream issue: https://github.com/sgl-project/sglang/issues/38815

Mirror issue: https://github.com/amdpilot-org/sglang/issues/563

## Finding

Recommendation: **accept**, with `fully_resolves_original: false` because the original H200/Rust/Inkling deployment could not be rerun here.

The base independently reproduced the precise defect: SWA returned branch length 96 while the insertion carried Mamba state from depth 192. The candidate's guard makes Mamba's checkpoint depth own the hybrid insertion length. The passing regression inspects the real unified-cache tree and verifies checkpoint ownership at 64, 128, 96, and 192; it is not merely a startup or hit-count smoke.

No executable Python TreeCore counterexample was found. A neighboring Full+SWA-only regression confirms SWA branch capping remains enabled without Mamba. The broader Full/SWA/Mamba class completed successfully for all 39 applicable methods; 92 methods were fixture-gated.

## Environment and evidence

- Imported source: `/job/repo/python/sglang/srt/mem_cache/unified_cache/components/swa_component.py`
- Interpreter: `/tmp/amdpilot-repo-j-89f17af919ec/venv/bin/python`
- GPU: one AMD Instinct MI350X, Torch `2.11.0+rocm7.2`, HIP `7.2.26015`
- Preserved raw evidence: `/job/review-evidence/base/` and `/job/review-evidence/candidate/`
- Native rebuild: not applicable; the candidate changes Python and tests only.

Rust TreeCore remains unverified because its extension loader could not execute `cargo` and failed before the test body. The real `thinkingmachines/Inkling` checkpoint and original prompts were unavailable, so end-to-end token/logprob parity remains unverified. These are explicit review limitations, not evidence against the shared-logic fix.
