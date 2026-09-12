# NGRAM cached-state insertion invalidation

Upstream issue: https://github.com/sgl-project/sglang/issues/38129

Mirror issue: https://github.com/amdpilot-org/sglang/issues/798

## Finding

The prepared `main` checkout at `358c163250ad3b1f62939b01ce1314a0a31a0365`
still reproduced the report. A request whose cached suffix anchors were missing
before insertion continued incrementally from those negative results afterward.
The existing node-version checks only detected eviction/reuse, while
`trie_epoch_` changed only on reset. Neither mechanism detected an ordinary
insertion that created a formerly missing suffix or changed BFS/PROB ordering.

Inspection found the same narrow candidate on the unmerged fork branch
`1sgtpepper/sglang:fix/ngram-cached-miss` at commit `17b2508`. The correction
tracks trie growth separately and rebuilds only a `MatchState` containing a
cached miss when new edges have since been added. Live anchors remain reusable;
their recency/frequency traversal already reads current node data.

## Evidence

- `raw/original_reproducer.txt`: exact issue reproducer before the change;
  both BFS and PROB returned `[30, 70, 80, 90]` for the cached request and the
  process exited 1.
- `raw/fixed_original_reproducer.txt`: the same reproducer against the rebuilt
  JIT library; cached, fresh, and erased requests all returned
  `[30, 40, 50, 60]` in both modes and the process exited 0.
- `raw/fixed_focused_tests.txt`: regression plus independent empty/partial
  corpus, zero/one/two appended-token, leaf-extension, incremental, and
  eviction-state boundaries passed.
- `raw/full_ngram_corpus_tests.txt`: the complete NGRAM corpus unit file passed
  with 43 tests and 12 subtests.
- `raw/native_artifacts.txt`: records the rebuilt native JIT library under the
  private runtime directory, including
  `/tmp/amdpilot-repo-j-54c3ab6ab28d/jit-cache/gfx950/sgl_kernel_jit_ngram_corpus/build-8825ebe152d79ec6/deps-b00fe0722a1007f5/sgl_kernel_jit_ngram_corpus.so`.

## Scope and limitations

This defect and its supplied reproduction are CPU-only and do not involve a
model, HTTP server, semantic model output, or distributed execution. No GPU
kernel was executed. The native library was compiled in the assigned gfx950
environment, but that is not claimed as GPU execution or a model-serving test.
