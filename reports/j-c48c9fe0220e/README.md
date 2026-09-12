# Independent review of PR 2626

Reviewed candidate https://github.com/amdpilot-org/sglang/pull/2626 at exact commit
`080ee781e6e6e43d2b6588546d91341139ee8be8` against upstream issue
https://github.com/sgl-project/sglang/issues/31404.

The recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365` reproduced the
original HTTP failure on a deterministic tiny Llama server: bogus model names
returned 200 and were echoed for both chat and completions. The exact candidate
returned the required 404 `model_not_found` response for both endpoints while
valid served-model requests still executed and returned 200.

The candidate regression passed, and independent handler probes additionally
covered explicit empty strings, omitted fields, both request types, and the
pre-conversion gate. Pydantic rejects explicit JSON null as a non-string before
serving validation. Python imported the reviewed source from
`/job/repo/python/sglang/srt/entrypoints/openai/serving_base.py`.

The recommendation remains `unverified`, not `accept`, because the candidate
also changes the Rust model gateway and the prepared environment contains
neither `cargo` nor `rustfmt`. That path was source-reviewed but could not be
compiled or executed. No C/C++/HIP/FlyDSL source changed, so no FlyDSL rebuild
applied. The HTTP engine run used one gfx950 GPU with ROCm 7.2 and random tiny
Llama weights; it does not qualify full-model semantics, other architectures,
or distributed serving.

Raw evidence is retained outside the revision-switching checkout under
`/job/review-evidence/`.
