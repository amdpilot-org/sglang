# Investigation result: MiniMax H3 reference-audio forward context

The prepared base already contains the source correction for the reported bug.
No additional runtime change is justified.

## Source issue

- Upstream issue: https://github.com/sgl-project/sglang/issues/35447
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1381

## Evidence

The report's failing call is `audio_vae.pre_block(...)` during
`minimax_h3_encode_reference_audio_rows`.  On this base, that helper wraps the
audio VAE preprocess, encoder, optional pre-block, and mean projection in:

```python
set_forward_context(current_timestep=0, attn_metadata=None)
```

The exact change was proposed in upstream PR
https://github.com/sgl-project/sglang/pull/35481 and subsequently landed in
merge commit `17313cf4b25d7420e1fd10b969d8b911d28e6498` through upstream PR
https://github.com/sgl-project/sglang/pull/35511.  That merge also added
`test_reference_audio_encode_sets_forward_context`, which reaches the
attention-bearing `pre_block` and asserts that timestep zero is active.

On the assigned AMD Instinct MI355X (`gfx950:sramecc+:xnack-`), a controlled
variant which replaced only this context manager with a no-op reproduced the
original assertion verbatim.  Restoring the checked-in implementation produced
an independently checked `(8, 32)` all-ones result (sum `256.0`,
`ref_audio_t=4`).  Additional boundary checks confirmed that a pre-existing
outer context is restored and that the temporary context is removed after both
successful execution and an exception.  Raw output is retained under `raw/`.

## Limitations

MiniMax H3 model weights and the report's reference image/video/audio inputs
were not provided, so the full model/server ref2va request was not reproduced
on this host.  The GPU fixture validates the precise audio-VAE execution and
context lifetime that caused the traceback; it does not validate model
semantics, generated media quality, distributed execution, or a full serving
request.  No native code changed, so no native rebuild was applicable.
