# Investigation of diffusion Qwen-Image SP=4 host OOM

Upstream issue: https://github.com/sgl-project/sglang/issues/36853

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1077

## Result

The prepared `main` source already contains the narrow loader correction that
addresses the reported transient duplication. Upstream PR
https://github.com/sgl-project/sglang/pull/34064 (merge commit
`8ba9385097322875f1a1c22e5f88356ebed2ec2a`) changed ordinary unquantized
TP=1 CPU loading so compatible checkpoint tensors become parameter storage
instead of allocating and copying a second tensor. Its published Qwen-Image
measurement with a CPU-offloaded DiT was 62.78 s to 34.67 s startup with the
same 9362 MiB peak GPU memory. Pure sequence parallelism replicates the DiT
and retains TP=1 parameter layouts on each rank, making this the relevant path
for the issue's suspected per-rank transient host copies.

The current regression
`TestOrdinaryWeightLoading.test_tp1_unquantized_linear_adopts_cpu_checkpoint_storage`
checks both pointer identity and numerical equality. The prepared source also
contains later rank-local TP/FSDP loading and host-spill work, but host spill is
gated to unified/shared host-device memory and is not claimed as the solution
for the report's discrete RTX 6000 Ada GPUs.

## Validation

- `python -m pytest -q python/sglang/multimodal_gen/test/unit/test_fsdp_load.py`
  passed 18 tests. This includes CPU checkpoint-storage adoption and two real
  GPU boundary cases.
- Three storage boundary tests passed independently on the assigned gfx950:
  CPU TP=1 adoption shares storage; ordinary GPU loading preserves its copy;
  explicit direct GPU loading adopts compatible GPU checkpoint storage. Each
  path also checks tensor values with `torch.testing.assert_close`.
- `python -m pytest -q python/sglang/multimodal_gen/test/unit/test_host_spill.py`
  passed 5 tests, covering file-backed fusion, reuse, disk-full fallback, and
  checkpoint fingerprint invalidation.

Raw command output and issue/PR snapshots are under `raw/`.

## Limitations

This job supplied one AMD Instinct MI355X (`gfx950`), not four 48 GB NVIDIA RTX
6000 Ada GPUs, and the Qwen/Qwen-Image weights were not available. Therefore
the exact four-rank host-OOM serving command, its peak host memory, NVIDIA
behavior, and end-to-end image semantics were not reproduced here. The result
is `candidate_verified`, not a claim of full-model or multi-node reproduction.
