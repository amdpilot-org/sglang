# Independent review of PR 830 at `3d10c70a56be17ed89300d26ffb77faf67ad992e`

Recommendation: **request changes**. The patch is a real partial fix, but it does not fully resolve the original host-wide ordering contract.

On the recorded base, the live-reading formula reproduces the reported false rejection. The candidate correctly prevents that race when every co-located rank belongs to the same TP group: its unit regression passed and a real three-process Gloo run gave every rank the same minimum-derived budget.

The remaining counterexample is multiple TP groups on the same host. `ranks_per_host()` divides the budget by all ranks on the host, while `host_memory_sync_group()` synchronizes only the current TP group. In the independent four-process Gloo case, two ranks in group A sampled 100 GiB available and received 22.5 GiB each; two ranks in group B sampled after 40 GiB was notionally consumed and received 12.5 GiB each. Four 20 GiB pools fit in the original 100 GiB with the 10 GiB reserve, yet group B is rejected. Thus ordering independence is narrowed to one TP group rather than established host-wide.

The exact reported 8xH200/TP8/model workload was unavailable. The assigned hardware was one AMD Instinct MI350X (`gfx950`), so no H200, full-model, or multi-node claim is made. The changed path is Python-only; the interpreter loaded `/job/repo/python/sglang/srt/mem_cache/pool_host/base.py`. No native source changed and no native rebuild was applicable.

Raw outputs and runnable independent cases are retained in this report directory.
