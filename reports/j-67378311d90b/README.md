# Chunked-prefill radix ownership investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/38319

Mirror issue: https://github.com/amdpilot-org/sglang/issues/510

The base already includes an allocator ownership fix: every grouped free clones
an incoming tensor view before the caller can mutate its backing request-table
row. The added regression covers the reported page size of 64 with the real
`RadixCache` and `PagedTokenToKVPoolAllocator`. It follows the C++ radix ordering:

1. a canonical page is present in the radix tree;
2. a duplicate request page is queued through a request-table view;
3. the row is rebound to the canonical radix page;
4. the grouped free is committed.

Temporarily replacing the ownership clone with `return free_index` makes the
test fail because the canonical radix page is released and the duplicate page
is leaked. Restoring the base implementation makes both the existing token
allocator regression and the new paged allocator regression pass.

Raw logs are retained under `raw/`. The GPU log additionally checks float16 KV
payload data against an independently constructed numerical reference.

The DGX Spark SM121/QSA/Qwen3.8-Flash-Next serving configuration was unavailable.
Consequently, the impossible first token (`248319`) and full serving behavior
remain unverified; this result verifies the concrete ownership candidate only.
