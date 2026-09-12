# Independent review of PR 1566

Reviewed `https://github.com/amdpilot-org/sglang/pull/1566` at exact commit
`0f30ee814e66ff26cdfe3bd990856182e25f2d51` against upstream issue
`https://github.com/sgl-project/sglang/issues/35673` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/1602`.

## Finding

The candidate is a valid narrow fix for one observed synchronization point, but
it is not a full reproduction or resolution of the original readiness hang.
Gemma 4 vision inputs at this call site are dense, so forwarding the host
`seq_len` as `max_seqlen` correctly avoids deriving the same value from a GPU
tensor with `.item()` in every encoder block. The exact candidate test failed
twice on recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`
because the keyword was absent, then the candidate's focused suite passed 9/9.

On the assigned single AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), direct
Triton-backend comparisons reproduced the previously reported large errors for
shapes `(3,5)` and `(4,9)` when `softmax_scale=1.0` was omitted from `forward`.
Passing the scale as the real Gemma call does reduced maximum absolute error to
`0.00195312` for both. Independent boundary shapes `(1,1)`, `(2,17)`, and
`(7,3)` had maximum absolute errors `0`, `0.00195312`, and `0.00195312`.
Thus that numerical counterexample does not justify another kernel change.

## Scope and limitations

Only one gfx950 GPU was available, whereas the report uses eight Radeon AI PRO
R9700 GPUs. Gemma-4-31B-it weights were unavailable. TP=8 collectives, the
reported topology, hybrid SWA allocation, the full multimodal/model path, and
HTTP `/health` reaching 200 were not exercised. Removing the host `.item()`
wait does not prove that an earlier GPU kernel or collective cannot remain
stalled; it can merely move or remove the point where such a stall is observed.

No native source changed. The imported candidate module was
`/job/repo/python/sglang/srt/models/gemma4_vision.py`; therefore no native
rebuild was applicable. The environment used Python from
`/tmp/amdpilot-repo-j-05e3123ba265/venv/bin/python`, PyTorch
`2.11.0+rocm7.2`, and HIP `7.2.26015`.

Raw outputs are retained under `reports/j-05e3123ba265/raw/`.
