# Investigation evidence

The official `zai-org/GLM-5.3-Flash` configuration declares `mhc: true`,
`hc_mult: 4`, `hidden_size: 4096`, and one bundled NextN prediction layer. It
does not declare an EAGLE3 `eagle_config`. In SGLang, `NEXTN` is an alias for
the plain `EAGLE` runtime; only `EAGLE3` enables auxiliary hidden-state capture.

On the recorded base implementation, invoking GLM5's default EAGLE3 capture
selected layers `[2, 22, 42]`. Each captured between-layer mHC tensor retained
shape `[2, 16384]`, so packing three layers produced width `49152`. The bundled
NextN implementation constructs `eh_proj` from `2 * hidden_size` and passes one
`hidden_size`-wide previous hidden state to `fused_eh_norm`; its expected
previous-hidden width is therefore `4096`.

The current repository's GLM-5.3-Flash serving recipes and H200 regression use
`EAGLE`/`NEXTN` for the bundled MTP head. Those paths do not request auxiliary
captures and instead pass the final, already-contracted target hidden state.
Consequently, choosing and contracting an arbitrary auxiliary layer would only
remove the shape error without evidence that the bundled head was trained for
that representation. The implemented correction rejects the invalid bundled
head plus EAGLE3 combination before model/kernel startup and leaves a distinct,
separately trained EAGLE3 draft checkpoint permitted.

No GLM-5.3 weights or 4x H20 system were available. The single assigned gfx950
was used only to confirm that `hc_contract([2, 16384], hc_mult=4)` exactly
matches an independent reshape-and-mean reference and returns `[2, 4096]`.
