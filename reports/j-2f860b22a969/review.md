# Independent review of PR 868

Candidate: https://github.com/amdpilot-org/sglang/pull/868  
Exact candidate commit: `3dfa1169b0945e725823522935ca98e320f9b8c2`  
Upstream issue: https://github.com/sgl-project/sglang/issues/38129  
Mirror issue: https://github.com/amdpilot-org/sglang/issues/897

## Recommendation

Accept. The candidate fully resolves the original CPU NGRAM cached-request
contract in the reproduced scenario and in independent stateful differential
coverage. No remaining counterexample was found.

## Evidence

The prepared checkout was already at the required recorded base
`358c163250ad3b1f62939b01ce1314a0a31a0365` on
`amdpilot/j-2f860b22a969`; there was no difference from the image-prepared
checkout.

On that base, the issue's exact reproducer exited 1. BFS and PROB both returned
`[30, 70, 80, 90]` for the cached request while fresh and erased-state queries
returned `[30, 40, 50, 60]`.

I then detached at the exact candidate commit. With `PYTHONPATH=/job/repo/python`,
imports resolved to the checkout, including:

- `/job/repo/python/sglang/__init__.py`
- `/job/repo/python/sglang/srt/speculative/cpp_ngram/ngram_corpus.py`
- `/job/repo/python/sglang/kernels/ops/speculative/ngram_corpus.py`

An empty candidate-only `SGLANG_JIT_CACHE_DIR` forced a native rebuild. The
candidate C++ source was compiled into:

`/tmp/amdpilot-repo-j-2f860b22a969/jit-candidate/gfx950/sgl_kernel_jit_ngram_corpus/build-20c863fe296177d5/deps-b5c94ba90328058a/sgl_kernel_jit_ngram_corpus.so`

The generated Ninja file records `c++ -std=c++20 -O3` compiling the checkout's
`trie.cpp` and linking against the prepared `libtvm_ffi`. The build directory is
keyed as `gfx950`; this particular extension contains CPU C++ sources and did
not compile or execute a GPU kernel.

The exact issue reproducer then exited 0 and returned `[30, 40, 50, 60]` for
cached, fresh, and erased-state queries in both BFS and PROB. The candidate's
focused test file passed: 43 tests and 12 subtests.

As an independent adversarial check, a deterministic differential test compared
one reused request state against a new request ID after each of 80 rounds per
mode. Rounds mixed newly learned continuations of the active history, unrelated
trie growth, unchanged and extended queries, and bounded-capacity eviction. The
candidate passed all 160 cached/fresh comparisons. The same test on the recorded
base failed at BFS step 0: cached `[20, 0, 0, 0]` versus fresh
`[20, 1000, 2000, 3000]`.

Raw commands, outputs, exit statuses, the candidate diff, and the independent
test script are retained outside the revision-switching checkout at
`/job/review-evidence-j-2f860b22a969/`.

## Code assessment

The defect is a cached null anchor: insertion can make it matchable without
changing any live node version. The candidate increments a trie growth epoch
when a new edge is created, records it in request match state, and rebuilds only
when a state contains a cached miss and the trie has grown. Existing live-anchor
validation and eviction epochs remain in force. This directly addresses the
original invalidation gap and preserves the incremental fast path when all
anchors are live.

## Limitations

The original issue is a model-free CPU data-structure bug, so no model, HTTP
server, semantic-accuracy workload, or distributed workload was relevant or
run. The host is x86_64 with one visible AMD Instinct MI350X (`gfx950`) and
PyTorch `2.11.0+rocm7.2`, but `gpu_execution` is false because the tested NGRAM
extension and reproducer execute on CPU. No claim is made about other model
architectures, serving behavior, multi-node execution, or GPU kernels.
