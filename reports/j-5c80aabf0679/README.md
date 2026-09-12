# Independent review of multimodal caller cache IDs

Upstream issue: https://github.com/sgl-project/sglang/issues/38651

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3411

Candidate: https://github.com/amdpilot-org/sglang/pull/3409 at exact commit
`e8cac2caf689119bd8c71eb7d7c7f0d960731448`.

## Recommendation

`request_changes`. The candidate is an honest partial implementation/report,
but it does not fully resolve the original feature contract. It adds and tests
caller-ID parsing and alignment, and its generic artifact cache can skip source
snapshotting and preprocessing on a trusted hot hit. Operational forwarding is
limited to the Kimi K3 image processor.

Independent execution reproduced two media-load boundary crossings for two
requests carrying the same video cache ID through Qwen VL, and two crossings
for the same audio cache ID through Whisper. This occurs on both the recorded
base and exact candidate. Source inventory finds only Kimi K3 calling
`prepare_media_artifacts`, so other image processors do not receive the same
caller-ID artifact behavior either.

Responses `previous_response_id` reconstructs historical messages from
`msg_store`; the candidate adds no stored artifact-reference mechanism. A Kimi
K3 image history may benefit from the process-local artifact cache, but no
compatible multimodal model and weights were available to qualify that serving
path, eviction fallback, or any video/audio history.

## Evidence

The prepared Python interpreter imported SGLang source from `/job/repo/python`.
Torch was `2.11.0+rocm7.2` with HIP `7.2.26015`. AITER imported its prebuilt
module from the prepared private runtime. The candidate changes no native
C++/CUDA/HIP/FlyDSL source, so a native rebuild was not applicable.

Commands run against the exact candidate:

```bash
PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-5c80aabf0679/venv/bin/python \
  /job/review-evidence-j-5c80aabf0679/reproduce_ignored_cache_ids.py

PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-5c80aabf0679/venv/bin/python -m pytest -q \
  test/registered/unit/managers/test_mm_hashes.py \
  test/registered/unit/entrypoints/openai/test_protocol.py \
  test/registered/unit/parser/test_jinja_template_utils.py \
  test/registered/unit/multimodal/test_media_artifact_processor.py

PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-5c80aabf0679/venv/bin/python \
  /job/review-evidence-j-5c80aabf0679/adversarial_artifact_cache.py
```

Results were respectively exit 0 with two video and two audio load attempts,
exit 0 with `94 passed, 27 subtests passed`, and exit 0 proving that the generic
trusted-ID artifact-cache primitive itself skips snapshot/preprocessing on a
hot hit. Raw outputs and scripts are retained under `evidence/`.

## Limitations

No compatible multimodal model weights were available and no GPU model
execution was performed. The supplied tiny Llama fixture validates transport
and engine execution only and cannot qualify multimodal decoding,
preprocessing, cache reuse, model-specific recomposition, or Responses history.
Consequently Kimi K3 end-to-end behavior and Responses serving behavior remain
unverified rather than failed. Video and audio failures are deterministic at
the pre-I/O boundary and require no weights.
