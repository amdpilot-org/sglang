# Independent review of overlap-spec candidate b6b8651

Upstream issue: https://github.com/sgl-project/sglang/issues/11762

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3017

Candidate PR: https://github.com/amdpilot-org/sglang/pull/2980

## Recommendation

Request changes. The candidate is useful test-only hardening for one NGRAM
overlap relay helper, but it changes no implementation or native source and
does not implement the original open feature roadmap. The tested behavior was
already present and passed an independent GPU-staged relay case at the recorded
base commit, so there is no failing-before/passing-after candidate fix.

The candidate's focused four-case regression passes. Independent zero-accept,
full-fixed-row, context-truncation, and GPU-to-host staging cases also pass.
The unrelated speculative KV-index grid passes on the assigned AMD Instinct
MI350X (gfx950), but it does not establish NGRAM end-to-end serving or the other
unchecked roadmap features.

## Native and environment evidence

Python imports `sglang` from `/job/repo/python/sglang`, while `sgl_kernel`
comes from the pinned
`/opt/venv/lib/python3.12/site-packages/sglang_kernel-0.4.6.post1-py3.12-linux-x86_64.egg`.
The candidate changes no native files, so no rebuild is applicable. A direct
operator test fails because that egg exposes neither
`reconstruct_indices_from_tree_mask_cpu` nor its GPU counterpart. This blocks
end-to-end NGRAM overlap qualification in the prepared stack. The environment
provides one MI350X GPU with ROCm/HIP 7.2 and Torch 2.11.0+rocm7.2; no EAGLE
model, distributed topology, EP/DeepEP setup, PD-disaggregation setup, or LoRA
model was supplied.

Raw logs and the exact candidate diff were retained outside the checkout under
`/job/review-evidence-j-a3057e6baf24/` while revisions were switched.
