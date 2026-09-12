# Investigation: transformers import compatibility

Upstream issue: https://github.com/sgl-project/sglang/issues/38183

Mirror issue: https://github.com/amdpilot-org/sglang/issues/790

## Result

The reported failure does not reproduce at the prepared `main` base
`358c163250ad3b1f62939b01ce1314a0a31a0365`, and the source already contains
the relevant collision protection.

With the pinned `transformers==5.12.1`, both `PreTrainedConfig` and
`PretrainedConfig` resolve to the same class. `import sglang` and direct imports
of all five modules named in the report succeed. With an isolated
`transformers==5.16.1` overlay and its matching dependencies, those imports
also succeed. The `qwen3_asr` module can be reloaded without a registration
collision because both registrations currently pass `exist_ok=True`.

No source correction was justified by the observed behavior, so this PR adds
only the investigation record and raw evidence.

## Reproduction commands

Pinned environment:

```bash
PYTHONPATH=/job/repo/python \
  /tmp/amdpilot-repo-j-ebdbe973c110/venv/bin/python <focused import script>
```

The script imports `sglang`, imports each reported module directly, verifies
that the two Transformers config spellings are identical, and reloads
`sglang.srt.configs.qwen3_asr`. Full output is in
`raw/import-transformers-5.12.1.log`.

Transformers 5.16.1 boundary:

```bash
/tmp/amdpilot-repo-j-ebdbe973c110/venv/bin/python -m pip install \
  --upgrade \
  --target /tmp/amdpilot-repo-j-ebdbe973c110/transformers-5.16.1 \
  'transformers==5.16.1'
PYTHONPATH=/tmp/amdpilot-repo-j-ebdbe973c110/transformers-5.16.1:/job/repo/python \
  /tmp/amdpilot-repo-j-ebdbe973c110/venv/bin/python <focused import script>
```

Full output is in `raw/import-transformers-5.16.1-boundaries.log`. The isolated
overlay includes 5.16.1's matching dependencies, including its newer required
`tokenizers`, without altering the prepared pinned environment.

## Limitations

- Testing used the prepared Linux/ROCm Python environment, not either reported
  Apple Silicon system. macOS package resolution and MLX execution remain
  unverified.
- This is an import-path issue and no GPU computation is involved. The assigned
  gfx950 GPU was therefore not used, and no model-serving or numerical claim is
  made.
- `pip check` in the prepared environment reports pre-existing TileLang
  dependency inconsistencies; the focused imports still pass. See
  `raw/pip-check-pinned.log`.
