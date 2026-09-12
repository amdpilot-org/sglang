# Investigation report

Upstream issue: https://github.com/sgl-project/sglang/issues/32158

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2123

Base commit: `358c163250ad3b1f62939b01ce1314a0a31a0365`

The prepared source still contained the reported mismatch. `SchedulerLoadInquirer.get_loads()` read only `self.running_batch`, while `SchedulerPPMixin.event_loop_pp()` rebinds that pointer to one entry of `running_mbs` for each microbatch slot. The issue timeline had no linked corrective pull request or commit when inspected.

The failing-before regression models four active requests distributed across PP slots and observes that the base implementation has no way to inspect those slots. The correction changes the load-inquirer callback to return all `running_mbs` in PP mode and a singleton `running_batch` otherwise. It counts unique `rid` values so a request referenced by more than one slot is not double-counted.

The available machine exposed one AMD Instinct MI350X (`gfx950`) GPU. The reported TP=2, PP=4 topology requires eight GPUs, so the original HTTP workload was not reproduced and no claim of multi-GPU or full-model validation is made. This change affects CPU-side scheduler bookkeeping only; no native rebuild or GPU numerical test was relevant.

Raw evidence is retained under `evidence/`, including the failing-before and passing-after pytest output, final pre-commit output, and GPU inventory.
