# BWAP implementation evidence

This report accompanies the implementation of batch-wise adaptive pruning from
https://github.com/sgl-project/sglang/issues/35987 (mirror:
https://github.com/amdpilot-org/sglang/issues/2691).

The implementation was ported from the proposal commit
`aba3237cbad0c796c56701ad43fdb68f9b8af147` onto prepared base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The server-argument integration
was adapted to the newer namespace-based configuration system. Tests were also
made genuinely CPU-capable by using a pure-PyTorch `SiluAndMul` test double that
still participates in the production type-based hook discovery.

Raw unit, GPU numerical, server, request, response, graph-capture, and fixture
records are under `evidence/`. Runtime model weights and caches remain outside
the repository at `/tmp/amdpilot-repo-j-da4cd9bbb840/`.

The result is classified `candidate_verified`: core computation and serving
paths are verified, but production-model quality and throughput are not.
