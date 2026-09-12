# Qwen3.5 GDN AWQ loading investigation

The prepared source contained tuple-aware loading for split Qwen3.5 GDN
projections, but it only bound ordinary/FP8 parameter names. AWQ Marlin creates
`qweight`, `qzeros`, and `scales`, so those parameters bypassed the wrapper.
Further, `qweight` and `qzeros` store eight 4-bit output values per `int32` and
must be split in packed checkpoint units.

The correction binds the AWQ parameters and uses the parameter's existing
packing adjustment for output-packed tensors. Group scales retain logical output
widths, and tensors packed on the input dimension are unchanged.

The focused regression failed before the source correction and passes after it.
An independent gfx950 check generated logical int4 weights, packed and split
them through the corrected loader, unpacked each shard, and compared GPU matrix
multiplication against the original logical tensor exactly.

This is a verified candidate for the identified loading defect, not a full-model
reproduction. The 27B checkpoint was absent, and the available AMD gfx950 cannot
qualify the reporter's CUDA sm_120 AWQ Marlin serving path or semantic output.
Raw issue snapshots, test output, GPU evidence, and model-availability evidence
are retained under `raw/`.
