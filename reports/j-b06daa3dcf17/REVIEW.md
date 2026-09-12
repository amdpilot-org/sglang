# Independent review of amdpilot-org/sglang PR 2901

- Candidate: `ad0162d0c4c667916f85904b30cd356d38feee84`
- Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`
- Recommendation: **request changes**
- Classification: **partial fix**

## Finding 1: unsafe fallback leaves the original contract unresolved

The normal candidate path correctly caps reported total memory to
`torch.mps.recommended_max_memory()` and available memory to the smaller of
system availability and remaining Metal headroom. However, when that API is
missing, returns an invalid value, or raises, `get_mps_recommended_memory()`
falls back to `psutil.virtual_memory().total`. The available-memory helper then
also falls back to uncapped host memory.

In the independent adversarial case (64 GiB host total, 56 GiB host available,
48 GiB Metal limit), both an absent API and a raising API returned 64 GiB total
and 56 GiB available. This exceeds the Metal limit and contradicts the issue's
required guardrail that SGLang must not report more than Metal can safely use.

## Finding 2: the added registered test fails repository validation

`pre-commit run --files ...` fails `check-registered-tests` because
`test/registered/unit/utils/test_mps_memory.py` calls `register_mlx_ci(...)`.
The repository taxonomy permits unit tests in this directory to register only
CPU suites.

## Verified behavior

The recorded base reproduced the original failure in all three production
paths: the SRT and multimodal available-memory paths returned 56 GiB, and the
device-property and multimodal total-memory paths returned 64 GiB.

At the exact candidate commit, its focused suite passed (10 tests), and an
independent integration simulation returned the expected 40 GiB available and
48 GiB total through all three reported production paths. Imports resolved to
the checked-out source under `/job/repo/python`; no installed SGLang copy was
used.

## Architecture and environment limits

The prepared node is Linux/x86_64 with Torch `2.11.0+rocm7.2` and MPS is not
built or available. Therefore the real Apple Metal value, actual UMA pressure,
paging prevention, and an Apple Silicon serving workload remain unverified.
AMD GPU execution would not validate MPS memory accounting, so no GPU test was
claimed. The candidate changes only Python and adds no native code; no native
rebuild was applicable.

Raw command output was preserved outside the revision-switching checkout in
`/job/review-evidence-j-b06daa3dcf17/`.
