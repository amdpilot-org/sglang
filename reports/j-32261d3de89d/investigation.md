# Investigation evidence

Upstream issue: https://github.com/sgl-project/sglang/issues/37457

Mirror issue: https://github.com/amdpilot-org/sglang/issues/955

Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared source still called `ServerArgs.resolved_dict()` from both startup
log sites, the HTTP `/server_info` handler (and `/get_server_info` alias), the
gRPC bridge, and the in-process engine readback. At the base, `resolved_dict()`
returned every one of the 495 fields without filtering. The failing-before
test invokes the actual HTTP handler and records all three credential sentinels
in its response.

Related-change review found the still-open upstream candidate
https://github.com/sgl-project/sglang/pull/37499. That candidate explicitly
deviates from the reported positive-allowlist requirement: it marks three known
fields secret and relies on credential-shaped names for future review. The
implementation here retains every current response key for compatibility but
uses an explicit allowlist for values. A newly introduced field therefore
appears as `<redacted>` until reviewed and added.

This newer base also publishes `launch_command`, which can contain the same CLI
credential even if `resolved_dict()` is safe. CLI and in-process Engine launch
descriptions now use the same allowlist decision, and runtime-context overlays
reapply it so a rotated secret cannot bypass the base projection.

Raw test output is retained beside this report. GPU execution and a native
rebuild are not applicable to this pre-model Python serialization defect.
