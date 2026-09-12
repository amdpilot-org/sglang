# GLM-5.2 pipeline-parallel IndexError investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/29162

Mirror issue: https://github.com/amdpilot-org/sglang/issues/2438

## Finding

The original failure was not reproduced, and this change intentionally does not alter runtime code. The exact reported source revision is `2cbe1e6404144dc753de01525a51cbd7a89c1fda`; the prepared base is `358c163250ad3b1f62939b01ce1314a0a31a0365`.

The issue requires GLM-5.2-FP8 weights, four NVIDIA B300 stages, `SGLANG_PP_LAYER_PARTITION=19,20,20,19`, and an intermittent production request. This job has one AMD MI350X (`gfx950`) and no GLM-5.2 weights or triggering request. Consequently, neither the model architecture nor the reported pipeline topology can be qualified here.

## Source evidence

At the reported revision, `event_loop_pp` copied each slot into scheduler-wide `self.running_batch`, `self.last_batch`, and `self.cur_batch`, called `get_next_batch_to_run()` through those implicit fields, and later paired `self.mbs[next_mb_id]` with the asynchronously received result. The reported exception means that pairing reached result processing with fewer sampled token IDs than requests.

Current source has extensive later changes in this area. In particular, PP planning now passes `running_batch` and `last_batch` explicitly and receives a plan containing `running_batch` and `batch_to_run`; launch takes the local `cur_batch`; and PP tensor dictionaries carry message types so proxy and output traffic are demultiplexed. These are relevant hardening changes, but no upstream change was found that identifies issue 29162 or demonstrates the GLM-5.2 PP=4 failure before and after. It would therefore be unjustified to label the source fixed based on code shape alone.

## Commands and raw outcomes

```text
$ python -m pytest -q test/registered/unit/managers/test_batch_result_processor_spec_grammar.py test/registered/unit/managers/test_batch_result_processor_hidden_states.py
13 passed, 2 subtests passed in 12.26s

$ python -m pytest -q test/registered/unit/managers/test_scheduler_recv_skipper.py test/registered/unit/managers/test_batch_result_processor_mamba_boundary.py
9 passed, 2 subtests passed in 12.36s

$ python -m pytest -q test/manual/chunked_prefill/test_scripted_pp.py
KeyboardInterrupt after 133.36s; exit 130; no completed test result

$ ROCR_VISIBLE_DEVICES=0 python -c 'import torch; ...'
Torch 2.11.0+rocm7.2
CUDA-compatible device API available: True
device count: 1
AMD Instinct MI350X
gfx950:sramecc+:xnack-
```

The two successful suites validate isolated result-processing and scheduler communication boundaries. The interrupted scripted fixture is weightless and simulated. None is a substitute for GLM-5.2 execution on four physical pipeline stages.

## Conclusion

Outcome: `not_reproduced`. A narrow code correction is not supported by issue-specific evidence in this environment. Reproduction remains dependent on the reporter's model weights/topology and, ideally, the exact request that triggers the intermittent mismatch.
