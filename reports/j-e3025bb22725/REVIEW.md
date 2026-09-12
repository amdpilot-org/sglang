# Independent review of amdpilot-org/sglang PR 1517

- Candidate: `54cdc6836ad6387df1b01d140353f15a4f73a94f`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **accept**

The recorded base reproduced the original accounting defect: with normal
`MemAvailable` capacity fixed at 2 GiB after the existing reserve and a
matching 12 GiB free hugetlb pool, the actual imported base implementation
reported only a 2 GiB host budget and rejected a 7 GiB request. The same
standalone contract passed at the exact candidate commit with a 12 GiB budget.

The candidate correctly limits hugetlb capacity to the default anonymous mmap
allocator and a recognized `SGLANG_HUGEPAGE_SIZE`, requires the configured page
size to match the kernel default pool, takes the larger of normal and hugetlb
capacity rather than adding alternative backing pools, and divides global
capacity across co-located ranks. Independent adversarial checks covered those
properties plus malformed/incomplete meminfo input. No source-level remaining
counterexample was found for the original contract.

The focused candidate regression completed with 8 passing tests both with the
prepared environment's plugin set and, independently, with external pytest
plugin autoload disabled.

No native source changed, so no native rebuild was applicable. Imports were
confirmed from `/job/repo/python/sglang`, not an installed SGLang wheel.

Architecture limitation: the assigned device is one AMD Instinct MI350X,
gfx950, with Torch 2.11.0+rocm7.2. GPU execution was not relevant to this CPU
host-memory accounting change and was not performed. The host has
`HugePages_Total: 0`, and changing node-wide hugepage reservations was outside
scope, so a physical hugetlb-backed allocation and full model/server
reproduction could not be run. This review verifies the original preflight
contract deterministically, not model semantics, HTTP serving, or a
distributed workload.

Evidence files in this directory contain the raw base failure, candidate pass,
focused pytest result, adversarial result, host meminfo, import paths, and GPU
identification.
