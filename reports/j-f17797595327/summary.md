# Correction generation 1

Candidate PR: https://github.com/amdpilot-org/sglang/pull/1247  
Independent review PR: https://github.com/amdpilot-org/sglang/pull/1337

At exact candidate commit `6b37f294aa085347e44d747b235dc4987c747823`,
`validate_ngram_capacity` used an `assert`. The recorded optimized-Python
reproduction accepted NGRAM capacity 4 with maximum trie depth 4 and printed
`OPTIMIZED_ACCEPTED`.

The correction uses an explicit `ValueError`, retaining the candidate's native
constructor guard, CLI documentation, and boundary tests. The identical
optimized-Python command now fails in the serving-argument validator. A
subprocess regression ensures this behavior is covered when assertions are
disabled.

Raw commands and outputs are retained in `raw/`. The freshly built native
library path is recorded in `raw/native_paths.txt`.
