# Evidence

- Prepared checkout: `/job/repo`
- Base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Interpreter: `/tmp/amdpilot-repo-j-26abe37b9fc7/venv/bin/python`
- Baseline output: `/tmp/amdpilot-repo-j-26abe37b9fc7/before-allowlist.txt`
- Focused test output: `/tmp/amdpilot-repo-j-26abe37b9fc7/test-qwen-language-model-only-final.txt`
- Broader test output: `/tmp/amdpilot-repo-j-26abe37b9fc7/test-server-args-model-config.txt`
- GPU/runtime probe: `/tmp/amdpilot-repo-j-26abe37b9fc7/gpu-environment.txt`

The broader suite's two failures are platform-expectation mismatches in context-parallel tests on ROCm. The focused regression and all repository pre-commit checks pass.
