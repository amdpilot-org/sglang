# Independent review of PR 1094

Candidate: https://github.com/amdpilot-org/sglang/pull/1094 at exact commit
`ee6033d403ad5790a1580c7a83c36d206cf01ec3`.

Upstream issue: https://github.com/sgl-project/sglang/issues/37848

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1126

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.

## Finding

Recommendation: **request changes**. The candidate is a verified partial fix,
not a full fix for the repository-wide original issue.

The original AST reproduction reports 53 quantization-aware construction sites
that pass `quant_config` without `prefix` on the recorded base. At the exact
candidate it still reports 38 and exits 1. The candidate fixes the four named
PR 1032 counterexamples and the inherited Bailing-linear/Phi cases, but leaves
independent live counterexamples in other architectures.

A constructor-level adversarial probe patches only Persimmon's heavyweight
dependencies and observes the real wrapper call. For both root and nested
outer prefixes, `PersimmonForCausalLM` omits the `ParallelLMHead` `prefix`
keyword. The nested expected value is `nested.lm_head`; omission defaults to
the empty string and cannot match that checkpoint-qualified name.

Other clear remaining source counterexamples include Persimmon's four linear
projections, Voxtral's two projections, Falcon-H1's QKV/output projections,
Hunyuan's `FusedMoE` experts and output head, and GPT-J's attention and output
head. The 38-site count also contains false positives such as literal
`quant_config=None`; therefore the count is evidence of incomplete coverage,
not a claim that every one of the 38 sites is independently proven defective.

## Commands and results

- Base independent audit: `53` sites, exit 1.
- Candidate independent audit: `38` sites, exit 1.
- Candidate retained four-counterexample AST check: `4/4`, exit 0.
- Candidate focused pytest suite: `18 passed, 42 subtests passed`, exit 0.
- Independent Persimmon wrapper probe: observed `prefix` omitted where
  `lm_head` / `nested.lm_head` was required, exit 1.

Raw command output was preserved outside revision switching in
`/job/review-evidence-j-33f7df6d7aa2/`. The prepared interpreter imported
`sglang` and the inspected model modules from `/job/repo/python/sglang`, not
from an installed wheel.

## Environment and limitations

The environment has one AMD Instinct MI350X (`gfx950`), ROCm 7.2, and PyTorch
2.11.0+rocm7.2. No GPU execution was necessary for the deterministic source and
quantization-resolver contract, and no affected architecture weights were
available. Thus no full-model, semantic-accuracy, serving, or distributed
workload claim is made. The candidate changes only Python, tests, and reports;
there is no native C++ change, so a native rebuild is not applicable.
