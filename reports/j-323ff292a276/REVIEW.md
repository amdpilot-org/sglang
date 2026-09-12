# Independent review of amdpilot-org/sglang PR 3203

Candidate reviewed: `5f2c2ea26a4a83bc9c13ed58a784124a881fcfe3`

Upstream issue: https://github.com/sgl-project/sglang/issues/18891

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3207

Candidate PR: https://github.com/amdpilot-org/sglang/pull/3203

Parent candidate PR: https://github.com/amdpilot-org/sglang/pull/3107

Parent independent review PR: https://github.com/amdpilot-org/sglang/pull/3161

## Recommendation

Request changes. The candidate is a substantial partial fix, and its final
commit fixes the concrete `BasevLLMParameter` subclass-loader failure reported
by the previous review. It does not fully satisfy the module-level comparison
contract because a checkpoint that omits live parameters is reported as a
successful match.

## Evidence

On the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, importing
`compare_module_weights_with_disk` fails because the feature is absent. The
trace is in `raw/base_original_failure.txt`.

At the exact candidate commit, all six candidate checksum unit tests passed.
The interpreter imported SGLang from `/job/repo/python/sglang`, including the
candidate's `weights_updater.py`; PyTorch was the pinned `2.11.0+rocm7.2`.
See `raw/import_paths.txt` and `raw/candidate_regression.txt`.

An independent GPU exercise used one AMD Instinct MI350X and the production
`ModelWeightParameter.load_column_parallel_weight` path. The loaded GPU value
equaled an independent CPU tensor reference, the comparison matched, and a
one-value GPU mutation produced a mismatch. This confirms the subclass fix is
effective beyond the candidate's simpler `BasevLLMParameter` regression.

The same adversarial script created a module with two live parameters and a
checkpoint containing only one. The included value matched, while the omitted
live value was deliberately `999.0`. `compare_module_weights_with_disk`
returned `match: true` with `parameter_count: 1`. At the worker/API layer this
becomes “All requested modules match the checkpoint.” This is a false positive
for a whole-module server/disk comparison and can conceal a truncated or
incomplete checkpoint. See `raw/adversarial_candidate.txt` and its reproducible
script `raw/adversarial_candidate.py`.

## Scope and limitations

No native source changed in the candidate, so no native rebuild was applicable.
Only one GPU was assigned; multi-rank TP/SP/DTensor behavior was not exercised.
The single-rank parallel-loader test fixed TP rank to zero without initializing
a distributed process group. FLUX.2-klein and Qwen-Image weights were not
available, so model-specific VAE/text-encoder serving and semantic validation
remain unverified. The tiny Llama fixture was not used because it cannot qualify
diffusion model architectures or semantics.

The candidate was not modified or merged. This branch contains review evidence
only.
