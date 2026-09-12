# Evidence

Candidate `4379f80a1126404096aed27bc3fbe37833ff1f04` was applied to recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365`. The retained audit exited 1 with
52 sites, reproducing PR 1032's central claim. The candidate's focused test
passed, confirming its Bailing linear fix was valid and should be preserved.

This correction qualifies the four explicitly named review counterexamples,
and incorporates the independently resolver-tested Phi correction from
sgl-project/sglang PR 39191. The focused quantization tests pass (18 tests,
42 subtests). The retained broad audit decreases to 38 but still exits 1.
Accordingly the result is deliberately `candidate_rejected`, not a claim that
the original repository-wide bug is fully fixed.

No GPU or model weights are required for the demonstrated prefix-routing
failure. No full-model architecture or semantic accuracy claim is made.
