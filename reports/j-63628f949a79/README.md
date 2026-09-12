# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/36776

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1097

`TestMMLU.test_mmlu` reads `accuracy_mmlu` and defaults to `0.00`. The five
reported DeepEP tests instead defined `accuracy_mmlu_threshold`, so their
configured thresholds were never consumed. A deterministic pre-fix fixture
confirmed that a score of `0.10` passed despite a nominal threshold of `0.61`.

The narrow correction renames those five attributes without changing their
numeric values. The regression checks the exact reported files and separately
checks rejection below a configured threshold, acceptance above it, and the
existing default behavior when no threshold is supplied.

Raw command output is retained in `raw/`. The full model tests were not run:
they require Ascend A3 hardware, an 8/16-NPU topology, and unavailable model
weights. The assigned gfx950 GPU is not a meaningful substitute for that
architecture-specific validation.
