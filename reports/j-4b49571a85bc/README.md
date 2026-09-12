# MiniMax-H3 `--model-variant` investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/33501

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1858

## Result

The reported `error: unrecognized arguments: --model-variant fl2va` is already
fixed at the reviewed base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`.
No source correction is justified.

The fix entered upstream in merged PR
https://github.com/sgl-project/sglang/pull/33275 (`[diffusion] model: support
minimax-h3`, merged 2026-08-02). Its server-argument patch adds both the
`model_variant` field and the `--model-variant` CLI option. The issue was opened
against an installation that did not contain that new source. Later issue
comments independently report that installing from source resolves the
unrecognized-argument error. PR https://github.com/sgl-project/sglang/pull/33365
is related but addresses automatic detection of bare local directory names; it
is not needed for the reported command because that command explicitly selects
`--model-type diffusion` and uses `MiniMaxAI/MiniMax-H3`.

## Reproduction and boundaries

Running the reported command through the prepared checkout accepted
`--model-variant fl2va`, logged `model_variant: "fl2va"`, and reached diffusion
worker startup. It did not reproduce the reported argparse error. Startup then
failed with `CUDA error: invalid device ordinal` on ranks 1 through 7 because
the job has one assigned AMD Instinct MI355X (`gfx950`) while the command asks
for eight GPUs. The launcher returned zero after worker shutdown, so the raw
log—not that process status—is the evidence for the environment boundary.

An independent parser check verified all of the following without loading
weights:

- the complete reported argument vector dispatches to `diffusion` and preserves
  `model_variant=fl2va`, `num_gpus=8`, and `ulysses_degree=8`;
- equals-form values `--model-variant=ref2va` and
  `--model-variant=hybrid` are accepted;
- a missing `--model-variant` value is rejected with argparse exit status 2.

Raw evidence is retained in `raw/`, including issue and PR metadata, the
historical server-argument patch, parser output, GPU inventory, and the exact
CLI startup log.

## Limitations

MiniMax-H3 weights were not available, and the assigned topology is one gfx950
rather than the reported eight-GPU deployment. Consequently this investigation
does not claim model loading, inference, semantic accuracy, numerical GPU
validation, or an eight-GPU/multi-node reproduction. The issue itself is only
the CLI rejection, which is resolved before those unavailable stages.
