# Investigation notes

- Upstream issue: https://github.com/sgl-project/sglang/issues/34482
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1587
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Source wrapper: `python/sglang/multimodal_gen/runtime/models/vaes/autoencoder_dc.py`
- Installed diffusers exposes `AutoencoderDC.enable_tiling(tile_sample_min_height=None, tile_sample_min_width=None, tile_sample_stride_height=None, tile_sample_stride_width=None)` and `disable_tiling()`.
- The recorded base and fetched mirror `main` both lacked those wrapper methods. The decoding stage had already replaced the issue snapshot's blanket exception swallow with an `AttributeError` warning, but an explicit tiling request still could not reach the inner Sana VAE.
- `failing-before.txt` preserves the issue-specific failure before the production change. All three cases raised `AttributeError` on the wrapper.
- The correction forwards both controls after lazy initialization and intentionally leaves Sana's default `vae_tiling = False` unchanged.
- The runtime-only GPU script is retained outside the worktree at `/tmp/amdpilot-repo-j-fe67bdd065a2/gpu_autoencoder_dc_check.py`; its output is retained in `gpu-numerical.txt`.
