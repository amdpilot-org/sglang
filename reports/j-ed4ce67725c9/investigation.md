# Independent review of PR 886

Candidate reviewed: `82ceba80bdfc5eed0c74e81ae5e3b402a0b09e0d`

The prepared checkout initially matched the required recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` exactly. The candidate was checked
out detached for testing, then this review branch was restored before writing
the report. Evidence was retained under the private runtime directory during
revision switches.

The base implementation expands the entire block-scale tensor to FP32 and
multiplies the entire FP8 shared-expert matrix in FP32 before casting to BF16.
On the assigned gfx950 GPU, a 4096x8192 conversion had a 2,657,092,096-byte
incremental allocation peak. The candidate processes scale-row-aligned chunks;
the same seeded input had a 1,337,459,200-byte incremental peak, a 49.66%
reduction, with identical packed-weight and E8M0-scale checksums.

The candidate's 10 focused tests passed. Independent adversarial comparisons
against a direct implementation of the old full-matrix algorithm also produced
byte-identical results for partial scale rows, multiple leading dimensions,
different block sizes, and grouped inputs. Invalid K alignment was rejected.

Recommendation: **accept** as a narrow, source-level memory reduction with
appropriate regression coverage. It validates and fixes a real excessive
allocation in the conversion named by the report. It does **not** establish
that the complete original deployment now fits in 121 GB, so
`fully_resolves_original` is false. The original failure also involves
checkpoint residency, concurrent H2D/load staging, unified CPU/GPU memory, and
two-node process behavior. An upstream issue comment independently identifies
concurrent weight-copy staging and suggests swap/thread limiting, reinforcing
that other peak contributors may remain.

Architecture limitations: only one AMD Instinct MI350X (`gfx950`) was assigned.
There was no NVIDIA GB10/SM120 device, second node, 156.31 GiB checkpoint,
CUDA 13 image, b12x preview backend, or original cookbook environment. Thus no
full model load, scheduler OOM, host RSS, NCCL/RoCE, b12x execution, or final
serving test was possible. The change is Python-only; no native source changed
and no native rebuild was applicable. Imports resolved to the checked-out
source under `/job/repo/python`, not an installed SGLang wheel.
