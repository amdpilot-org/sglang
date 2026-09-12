# Correction generation 2: multimodal caller cache IDs

Upstream issue: https://github.com/sgl-project/sglang/issues/38651

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3387

Candidate parent: https://github.com/amdpilot-org/sglang/pull/3322 at exact
commit `a6d1911d49dd87c3052eed49a000c2c6817b8154`.

Independent review parent: https://github.com/amdpilot-org/sglang/pull/3384.

## Result

The candidate's valid API parsing, identity derivation, Kimi K3 image artifact
cache, and native image/video/audio cache-ID alignment are preserved. The
review's remaining video and audio counterexamples were independently
reproduced against the exact candidate before it was applied here.

`evidence/reproduce_ignored_cache_ids.py` invokes representative supported
processor entry points twice with stable caller IDs. The Qwen video path calls
`load_mm_data` twice, and the Whisper audio path calls `load_audio` twice. The
script deliberately stops at the load boundary, so it needs neither media nor
model weights and measures the promised early no-I/O boundary directly:

```text
video load attempts with repeated cache_id: 2
audio load attempts with repeated cache_id: 2
```

The source inventory in `evidence/candidate/artifact_call_sites.txt` contains
one processor call site, in Kimi K3. Consequently, caller-ID artifact caching
does not cover images generally either. The Responses API retains historical
request content, but no compatible multimodal model weights were available to
demonstrate that `previous_response_id` avoids historical media loading,
decoding, preprocessing, and feature hashing. The deterministic tiny Llama
fixture cannot qualify multimodal behavior and was not used as substitute
evidence.

No further processor source was changed. Generalizing the artifact cache is not
a mechanical forwarding change: each processor must define a prompt-independent
artifact, modality-specific snapshot/decode behavior, preprocessing-key inputs,
and request recomposition. Doing that without the affected architectures and
weights would be a speculative source change expressly outside the evidence
available here.

## Reproduction

```bash
/tmp/amdpilot-repo-j-42014eeffebd/venv/bin/python \
  reports/j-42014eeffebd/evidence/reproduce_ignored_cache_ids.py

/tmp/amdpilot-repo-j-42014eeffebd/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_mm_hashes.py \
  test/registered/unit/entrypoints/openai/test_protocol.py \
  test/registered/unit/parser/test_jinja_template_utils.py \
  test/registered/unit/multimodal/test_media_artifact_processor.py

/tmp/amdpilot-repo-j-42014eeffebd/venv/bin/python -m compileall -q \
  python/sglang/srt
```

The focused suite passed with `94 passed, 27 subtests passed`. Raw output and
exit codes are retained under `evidence/`.
