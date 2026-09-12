# Independent review of PR 1734

Reviewed `https://github.com/amdpilot-org/sglang/pull/1734` at exact commit
`48f2160e1dfe73a357bd8d257a5b97ab13330fff` against upstream issue
`https://github.com/sgl-project/sglang/issues/34857` and mirror issue
`https://github.com/amdpilot-org/sglang/issues/1779`.

## Recommendation

**Accept.** The candidate fully resolves the remaining router-GEMM portion of
the original issue on the assigned MI350X/gfx950. The prepared base already
contains the GLM-5.2 correction-bias preservation fix, independently covered by
9 passing tests. No remaining numerical counterexample was found.

## Evidence

On recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`, the actual imported
`aiter_dsv3_router_gemm` returned bf16 for all five tested shapes. All values
were bf16-representable. Untuned shapes exactly equaled an independent fp32
reference rounded through bf16; the tuned production shape `(8,7168,256)` was
also bf16-only and had maximum fp32-reference error `2.295e-2`.

At exact candidate commit `48f2160e1dfe73a357bd8d257a5b97ab13330fff`, the candidate regression
passed. An independent probe covered production `(8,7168,256)`, fallback
`(65,512,64)`, odd/skinny `(1,513,63)`, degenerate-K `(2,1,3)`, and
single-output `(1,7168,1)` cases. Every output was fp32, every output element
retained precision beyond bf16, and maximum error against fp32-input
`torch.nn.functional.linear` was between zero and `7.153e-7`. The corresponding
bf16-rounded reference errors reached `7.798e-3`.

The source import resolved to the checked-out
`python/sglang/srt/layers/rocm_linear_utils.py`, not an installed SGLang copy.
The candidate changes Python only; no C++, FlyDSL, or other native source is
changed, so a native rebuild was not applicable. Aiter native modules used by
the base reproduction were loaded/built from the private runtime cache recorded
in the raw log.

## Limitations

Validation used one AMD Instinct MI350X (`gfx950`) with Torch 2.11.0+rocm7.2
and HIP 7.2.26015. No GLM-5.2 weights were available, so this is not a
full-model semantic/accuracy validation. No multi-GPU or multi-node run was
performed. The candidate bypasses tuned aiter for this router GEMM; its
performance relative to a tuned aiter entry was not benchmarked. That is a
performance risk, not a remaining counterexample to the reported fp32 output
and correction-bias contract.

Raw commands and outputs are retained under `raw/`.
