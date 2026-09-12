# Independent review of PR 3072

Reviewed candidate: `0489de0c3c17b59743e84a05620c0376113d4376`

Recorded base: `358c163250ad3b1f62939b01ce1314a0a31a0365`

Recommendation: **accept**. The candidate fully resolves the original documentation request within its stated scope.

## Findings

No blocking defect was found. The candidate expands the lone Slurm example into four useful patterns: single-node serving, corrected multi-node serving, offline inference, and Apptainer/Singularity. It also fixes the base example's important rank-expansion bug by evaluating `SLURM_PROCID` inside each `srun` task, uses distinct `%n` task logs, current CLI spellings, and a bounded health check.

The exact candidate test fails 5/5 cases on the recorded base and passes 5/5 at the exact candidate commit. Independent checks parsed all seven Bash blocks, compiled the embedded offline Python, and checked resource/rank/log/readiness invariants without relying on the candidate's prose.

This is a documentation-and-test change. It does not change native code or runtime import machinery, so a native rebuild is not applicable. The prepared interpreter imported SGLang from `/job/repo/python/sglang/__init__.py`.

## Environment boundary

The review host has one AMD Instinct MI350X (`gfx950`) with Torch `2.11.0+rocm7.2`, but it is not a Slurm allocation. `sbatch`, `srun`, Apptainer/Singularity, and Mintlify were unavailable. Therefore the review does not claim end-to-end scheduler execution, two-node NCCL/RCCL behavior, container passthrough, large-model loading, numerical correctness, or rendered-site validation. Those are retained as limitations rather than mislabeled as successes.

Raw command results are retained beside this report. The more detailed structured assessment is in `result.json`.
