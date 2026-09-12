# Selective DSA dequantization correction

Candidate [PR 2736](https://github.com/amdpilot-org/sglang/pull/2736) was
reproduced at exact commit `21e526df7844155adec0c2163992178bf899ba99`.
Independent review [PR 2799](https://github.com/amdpilot-org/sglang/pull/2799)
was also inspected and its three concrete scaling counterexamples were rerun on
the assigned AMD Instinct MI355X (`gfx950`).

Before correction, 131,072 prefix rows with top-k 2,048 measured 1.519x full
latency for 64 query rows, while the 256- and 1,024-query late fallbacks measured
1.993x and 2.291x. Their peak allocations were respectively 100,483,584,
166,724,096, and 213,910,016 bytes versus 150,994,944 bytes for full
dequantization. Raw output is in `evidence/candidate_adversarial_scaling.log`.

The correction adds a pre-deduplication gate based on the top-k entry count.
Inputs exceeding half of the full-prefix row count now choose full
dequantization before allocating remap intermediates or calling `torch.unique`.
After correction, the reviewed 64/256/1,024-query cases measured
0.795x/0.802x/0.827x of the separately warmed full baseline, all with exactly
150,994,944 bytes peak allocation. The candidate's beneficial 262,144-row,
8-query case remains selective and measured a 2.500x speedup with 4,641,408
bytes of compact BF16 staging versus 301,989,888 bytes for the full buffer.

The assigned AMD architecture can validate the Triton dequantization,
deduplication, remapping, and fallback behavior. It cannot validate BF16
`flashmla_sparse` attention output or end-to-end sparse-prefill performance,
which require NVIDIA Hopper/Blackwell. The repository's backend fixture also
stops on this ROCm setup because its page size is 64 while the available legacy
HIP DSA path requires page size 1; that failure is retained in
`evidence/pytest_dsa_fp8_prefill.log` and is not treated as evidence for another
source change.
