# TeaCache CFG lifecycle investigation

The prepared base (`358c163250ad3b1f62939b01ce1314a0a31a0365`) still contained both reported defects. The preserved pre-fix reproduction shows the serial negative call resetting `cnt` from 1 to 0 and a negative-only CFG-parallel rank retaining state into the next request.

The fix uses the current `forward_batch.is_cfg_negative` value for lifecycle decisions. Serial CFG resets on its positive call only; CFG-parallel execution resets each rank's locally owned branch. The same topology flag prevents `get_skip_boundaries()` from doubling timestep boundaries when only one branch executes locally.

Related upstream PR https://github.com/sgl-project/sglang/pull/24227 remains open and proposes counter arithmetic for the serial off-by-one symptom. It does not address request isolation or CFG-parallel boundary scaling. The source issue timeline contained no upstream fix linked at investigation time.

Evidence is retained under `reports/j-a18a6c35f2b5/raw/`. The GPU check used the assigned MI350X/gfx950 for GPU-resident cache tensors, but this was not a full model or distributed reproduction.
