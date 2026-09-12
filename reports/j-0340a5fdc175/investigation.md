# Investigation evidence

- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- The base called `lookup_kv(token_ids, req.rid)` while `key.cache_salt` was in scope and constructed `StoreMetadata` without `req.cache_salt`.
- The prepared interpreter reports `ModuleNotFoundError: No module named 'lmcache'`, so a live MP daemon reproduction was not available.
- LMCache issue `LMCache/LMCache#4825` remains open. Its companion PR `LMCache/LMCache#4842` is open and introduces `supports_cache_salt = True`, a salted `lookup_kv` argument, and `StoreMetadata.cache_salt`.
- SGLang draft PR `sgl-project/sglang#37229` is open and independently uses the same capability-gated/fail-closed design. This checkout did not already contain that solution.
- The focused regression was run before implementation and failed 4 tests; after implementation it passed 5 tests. The broader radix-cache run passed 44 tests and 61 subtests.
- Ruff was unavailable in the prepared interpreter (`No module named ruff`). Python compilation and `git diff --check` passed.
