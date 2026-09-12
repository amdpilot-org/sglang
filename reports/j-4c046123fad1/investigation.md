# Independent review of PR 1020

Upstream issue: https://github.com/sgl-project/sglang/issues/37457

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1054

Candidate: https://github.com/amdpilot-org/sglang/pull/1020 at `94dc025e62fcc32ea7f88cff0fad49a24dc97e49`

Recorded and prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365` (no difference).

## Finding

Recommendation: **accept**. The exact candidate fully resolves the original issue's stated contract.

On the base, an independent direct invocation of the actual async `http_server.server_info()` handler returned unique `api_key` and `admin_api_key` sentinels in cleartext. On the candidate, the handler, startup-log projection, CLI launch-command projection (both `--option=value` and split forms), internal-state projection, and an unknown future field all withheld the sentinels. The operational `ServerArgs` object retained the original credential values.

The candidate uses a positive allowlist: all 495 current `ServerArgs` fields were compared with the allowlist, 492 are explicitly public, and exactly `api_key`, `admin_api_key`, and `ssl_keyfile_password` are redacted. The allowlist contains no stale names. An unknown future field is redacted to the deterministic `<redacted>` marker unless its value is `None`, which preserves the existing unset-value shape.

The checked interpreter imported `sglang`, `server_args.py`, and the new `server_args_diagnostics.py` from `/job/repo/python`, so the tested source was the temporary exact-candidate checkout rather than an installed copy. The candidate changes no C++ or other native source, so a native rebuild is not applicable.

## Evidence and tests

Raw outputs were preserved outside the revision-switching checkout under `/job/review-evidence-j-4c046123fad1/`.

- Base direct handler reproduction: exit 0, and the assertions confirmed that the base response equaled the two cleartext sentinels. This is a successful reproduction of the defect, not a passing security test.
- Attempting to transplant only the candidate regression onto the base failed during collection because it imports the candidate-only diagnostics module. This was recorded separately and was not used as failing-before proof.
- Candidate regression class: `5 passed, 6 subtests passed`.
- Candidate related suite: `272 passed, 2 deselected, 70 subtests passed`. The two deselections are the same prepared-ROCm-inapplicable context-parallel cases documented by the candidate; they do not exercise diagnostic publication.
- Independent adversarial script: passed handler top-level and nested internal-state redaction, operational-value preservation, both CLI syntaxes, future-field default privacy, and startup-log projection.
- Source search covered HTTP, deprecated HTTP alias (which delegates to the same handler), in-process Engine, gRPC bridge, scheduler internal state, runtime overrides, and startup logging publication sites.

## Limitations

No model weights, full model server, sockets, distributed workload, or semantic model execution were used. The real async handler was invoked with a no-socket `TokenizerManager` fixture, which is sufficient for the pure-Python serialization contract but makes no model-serving claim. The assigned environment is ROCm with an intended single gfx950 GPU, but GPU execution is irrelevant to this pre-model disclosure and was not performed. No native code changed and no native rebuild was performed.

No remaining counterexample tied to the original startup-log or server-information contract was found.
