# Independent review of PR 3100

Candidate: https://github.com/amdpilot-org/sglang/pull/3100 at `c7ed070a5f7f0d414863f34edb901badf32e453e`

Upstream issue: https://github.com/sgl-project/sglang/issues/32485

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3124

Recommendation: **request changes**. The candidate is a partial original-issue fix, not merely test hardening: it implements pre-stream shard pruning for both requested recognized layouts and repairs the exact `model.nextn.layers.*` mixed-layout failure reported against its parent. However, it still violates the original requirement that an unknown layout safely fall back to all prepared shards.

## Independent findings

At the recorded base, a recognized native MTP index still sent all five prepared shards to the RunAI iterator, reproducing the original unnecessary-open behavior.

At parent candidate `e9952a5825d098de1c068d061062fe2522981d27`, the exact reviewed mixed case sent only `draft-1.safetensors`, dropping the shard mapped by `model.nextn.layers.1.shared.weight`.

At reviewed candidate `c7ed070a5f7f0d414863f34edb901badf32e453e`, the focused suite passed and that exact case safely retained all prepared files. Two independent mixed-layout cases still failed:

- `mtp.1.decoder.weight` plus `model.mtp_layers.1.shared.weight`
- `mtp.1.decoder.weight` plus `model.nextn_predict_layers.1.shared.weight`

In each case every prepared filename appeared in the index, but only `draft-1.safetensors` was passed to RunAI. The unknown draft shard was silently dropped. `model.mtp_layers.*` is not hypothetical naming invented solely for the review: the repository contains model handling/tests for `model.mtp_layers.0.*` keys.

The cause is the unknown-layout detector `(?:^|\.)(?:mtp|nextn)(?:\.|$)`, which sees only exact dot-delimited `mtp` or `nextn` components. Names such as `mtp_layers` and `nextn_predict_layers` bypass the fallback and are then treated as ordinary target keys.

## Environment and scope

The candidate changes Python source only. The loaded source path was `/job/repo/python/sglang/srt/model_loader/loader.py` under the pinned interpreter. No native source changed, no native rebuild target is configured, and a native rebuild was therefore not applicable.

The prepared machine is x86_64 with one AMD Instinct MI350X, Torch 2.11.0+rocm7.2, and HIP 7.2.26015. GPU execution was not used because JSON classification and the exact filename list passed before tensor streaming are the contract under review; a GPU smoke would not validate it. Live object-storage I/O, production model semantics, and provider header-read counts remain unverified because credentials and a production sharded MTP checkpoint were unavailable.

Raw revision-stable evidence is retained outside the checkout at `/job/review-evidence-j-1de4e777e677/`.
