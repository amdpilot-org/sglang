# Investigation report: sglang#31588

The prepared base already contains the reported correction.  The automatic
hybrid-SWA KV sizing path reads `eagle_draft_num_layers`, adds every draft
layer to the per-token coefficient, and uses that combined target-plus-draft
coefficient in `calculate_pool_sizes`.  Multi-layer EAGLE creates one runner
per speculative step and passes the same resolved pool configuration to each
runner, so the eight built-in Inkling MTP layers are represented by the eight
draft layers in that coefficient.

I added an issue-specific regression using the report's 65.45 GiB budget,
eight draft layers, 0.1 SWA ratio, and page size 128.  It independently
reconstructs the old target-only heuristic: the target allocation fits, but
target plus all eight draft layers exceeds the budget.  The checked-out
implementation's selected capacity remains within the same budget.  A second
test covers the page-alignment boundary immediately below two pages.

This is arithmetic coverage of the actual pool configurator, not a claim of a
full server reproduction.  The reported 8x NVIDIA B300 TP8 machine and
Inkling-NVFP4 weights were unavailable.  The assigned AMD gfx950 GPU was used
only for an independent PyTorch/ROCm numerical check; that check does not
qualify the NVIDIA architecture, Inkling model, FP4/MXFP8 behavior, TP8, or
eight-GPU allocation sequence.

Raw logs are retained in `raw/`.
