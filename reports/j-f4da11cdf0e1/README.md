# Qwen preprocessing cache investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/1932

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3040

The prepared base already had an OpenAI `image_url.content_hash` extension, a
bounded CPU multimodal preprocessing cache, cache flush integration, and
single-flight behavior. The cache was only consumed by the Kimi-K3 artifact
path. Qwen2-VL—the architecture named in the report—always loaded the media and
ran its combined Hugging Face text/image processor.

This candidate wires exact Qwen request preprocessing into the existing cache.
With `--trust-mm-content-hashes`, a complete set of caller-provided SHA-256 IDs,
and CPU feature transport, the hot path resolves the cache entry before opening
the supplied URLs. The retained output is copied both on insertion and lookup
because scheduler processing mutates multimodal items.

## Scope and limitations

- Qwen's current Hugging Face call combines prompt tokenization and image
  preprocessing. The rendered prompt and ordered image IDs are therefore part
  of the key. Changed prompts are misses; image-only reuse across prompts is not
  implemented.
- Entries are process-local, bounded LRU state. They do not survive restart,
  eviction, or cache flush and are not shared between tokenizer workers.
- Video and audio requests are excluded. CUDA IPC/VMM feature transport is
  excluded to avoid retaining GPU-backed request outputs.
- Caller hashes are trusted only with the explicit server flag. A false hash can
  select stale/wrong preprocessing, which is why the default verified path does
  not bypass media reads.
- No Qwen2-VL model weights were available in the prepared runtime. The supplied
  tiny Llama fixture cannot qualify a different architecture or multimodal
  preprocessing, so no serving/GPU numerical claim is made.
- No native code changed, so a native rebuild was not applicable.

Raw logs are under `raw/`. `baseline-regression.log` runs the new regression
against an archive of recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`
and records three expected failures. `focused-tests.log` records the same tests
plus related API, parser, hash, and cache suites passing on the candidate.
