# Independent review of PR 1745

Candidate: https://github.com/amdpilot-org/sglang/pull/1745

Exact commit: `648a8992ef222042ffac9cf931e929a1d3600427`

Upstream issue: https://github.com/sgl-project/sglang/issues/35564

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1780

## Recommendation

Accept. The exact candidate fully resolves the original issue in the deterministic parser contract exercised by the report. It also resolves the independently reported remaining Mistral counterexample: JSON arrays whose object-separator comma is followed by no space or by other JSON whitespace.

## Evidence

The prepared checkout was exactly the recorded base `358c163250ad3b1f62939b01ce1314a0a31a0365`; there was no image-prepared revision difference.

On that base, the original eight-parser reproduction exited 1. GLM, GLM45, GLM47, Minimax-M2, and Step3 disagreed with one-shot parsing. A separate Mistral no-space array driver showed that one-shot parsing returned three calls, while whole-increment and character-at-a-time streaming returned only the first call and leaked the remaining JSON into `normal_text`; all 122 possible two-way splits failed.

After temporarily checking out exact candidate commit `648a8992ef222042ffac9cf931e929a1d3600427`, imports resolved to the checkout (`/job/repo/python/sglang/...`), not an installed package. The original reproduction exited 0 for all eight parsers. The candidate's focused Mistral/parity tests passed 29 tests and 2,077 subtests. The complete function-call unit directory passed 551 tests and 2,098 subtests.

Independent adversarial testing covered all 64 pairs drawn from no whitespace, one/two spaces, tab, LF, CR, CRLF, and mixed whitespace after two separators. Inputs included empty arguments, nested objects/arrays, and a comma inside a JSON string. Whole, character, fixed-size, every two-way split, and deterministic randomized three-way delivery produced one-shot-equivalent calls with no leaked normal text in all 13,424 deliveries.

Raw command output is retained outside the revision-switching checkout at `/job/review-evidence-j-fff0162fc793/`.

## Scope and limitations

This is pure Python string/JSON parser behavior. The candidate changes no C/C++/HIP/CUDA/native source, so no native rebuild was required or performed. No GPU was used. The prepared environment is Python with PyTorch `2.11.0+rocm7.2` on a ROCm/gfx950 host, unlike the reporter's CUDA/RTX 4090 environment. No model weights, tokenizer, HTTP serving process, semantic model behavior, GPU kernel, multi-GPU, or distributed workload was exercised. Those limitations do not prevent verification of the reported deterministic parser contract, but no broader serving or architecture claim is made.
