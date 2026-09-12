# Runtime schedule-policy switching qualification

The recorded base did not accept a string `schedule_policy` request and did not
allow or apply that key in `Scheduler.set_internal_state`. This change implements
the contract proposed in upstream PR #31710 and adds stronger regression coverage
for state preservation, actual scheduling behavior, and concurrent HTTP updates.

The private engine run used one assigned AMD Instinct MI350X (`gfx950`) and the
deterministic tiny random Llama recipe inspected from amdpilot-org/sglang PR 649
at commit `f1d603677ca76a9ea21124a544e405c5b0cbd315`. Generated weights remain outside
the checkout at `/tmp/amdpilot-repo-j-326f6c074471/models/tiny-random-llama`.

Evidence:

- `evidence/unit/pytest.log`: request-schema regression.
- `evidence/unit/pytest-scheduler.log`: transition, rejection, identity, and ordering tests.
- `evidence/unit/pytest-policy-suite.log`: existing and new policy tests together.
- `evidence/http/http-probe.json`: accepted/rejected values, concurrent updates,
  generation, unchanged startup identity, and retained prefix-cache hit.
- `evidence/http/server.log`: full real-server log.
- `evidence/http/run-metadata.json`: command, process identity, and cleanup.
- `evidence/gpu-before.csv` and `evidence/gpu-after.csv`: assigned GPU evidence.
- `evidence/fixture-create.log`: private fixture identity and weight digest.

The synthetic fixture does not qualify semantic accuracy, GLM-5.2-NVFP4, TP8,
or the original B300 production-load measurements. Those remain unavailable.
