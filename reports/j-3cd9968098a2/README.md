# Independent review of PR 2784

Candidate: `e3da2b295dcd70b2c65158c3458006ea03f2b9b4`

The candidate is a useful partial fix: its weighted, non-blocking controller bounds work after a request reaches tokenizer-owned multimodal preprocessing, and its accounting/cancellation regression suite passes. It does not fully resolve the original CPU-OOM contract because FastAPI/ASGI body buffering, JSON parsing, and base64 retention happen before the controller acquires capacity.

Recommendation: **request changes**. The PR should either add admission before large request bodies are buffered/parsed, or narrow its claim and explicitly leave the original end-to-end feature open. It must not be described as the complete original-issue fix in its current form.

## Evidence

- The recorded base has no admission controller; importing it fails (`evidence/base-import-reproduction.log`).
- Imports at the candidate revision resolve to `/job/repo/python`, not an installed SGLang wheel (`evidence/candidate-import-paths.log`).
- The candidate's declared focused regression set passes: 454 passed, 2 platform-inapplicable HIP context-parallel tests deselected, 58 subtests passed (`evidence/candidate-declared-regression.log`).
- An unrestricted focused run has the same two unrelated HIP failures and 318 relevant passes (`evidence/candidate-focused-tests.log`).
- Independent adversarial reproduction with a configured budget of one retained 32 distinct, already-parsed 4 MiB payloads (134,217,782 bytes total; measured maximum-RSS growth 135,076 KiB) before all were rejected as busy (`evidence/adversarial-pre-acquisition-memory.log`). This is the request-body/base64 stage explicitly included in the original report.
- The candidate changes only Python/tests/reports; no native or build path changed, so no native rebuild applies (`evidence/native-scope.log`).
- The assigned node had one AMD Instinct MI350X (`gfx950`) with Torch 2.11.0+rocm7.2 / HIP 7.2. No GPU execution was needed or performed because the changed behavior is CPU admission/accounting and the exact Kimi-K2.6 workload and weights were unavailable (`evidence/environment.log`).

Upstream issue: https://github.com/sgl-project/sglang/issues/33035

Candidate mirror issue: https://github.com/amdpilot-org/sglang/issues/2731

Review mirror issue: https://github.com/amdpilot-org/sglang/issues/2817
