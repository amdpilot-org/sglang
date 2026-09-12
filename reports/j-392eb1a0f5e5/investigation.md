# Investigation evidence

- Upstream issue: https://github.com/sgl-project/sglang/issues/35433 (open, no comments when inspected on 2026-09-12).
- Mirror issue: https://github.com/amdpilot-org/sglang/issues/1385.
- Prepared base: `358c163250ad3b1f62939b01ce1314a0a31a0365`.
- Upstream `main` inspected at `6953dae0057b197cad4b665bba6692d2ce02516c`; its `encoding_dsv4.py` had no change from the prepared base for this behavior.
- The chosen behavior is explicit rejection rather than normalization. Moving inline system instructions to the front would change their precedence; rejection is narrow and is converted to HTTP 400 by `OpenAIServingBase.handle_request` before inference.
- Raw paths: source checkout `/job/repo`; interpreter `/tmp/amdpilot-repo-j-392eb1a0f5e5/venv/bin/python`; runtime/cache root `/tmp/amdpilot-repo-j-392eb1a0f5e5`.
