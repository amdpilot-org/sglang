# LPM anti-starvation aging investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/31954

Mirror issue: https://github.com/amdpilot-org/sglang/issues/3338

The prepared base still sorted LPM queues solely by matched-prefix length. Two
related upstream candidate PRs were inspected before implementation:

- https://github.com/sgl-project/sglang/pull/31723
- https://github.com/sgl-project/sglang/pull/33210

Both remained open. The current repository has since moved scheduling options
into `arg_groups/fields/schedule.py` and added HRRN, so the correction was
adapted to the current implementation. Unlike PR #33210, requests accrue LPM
age only while LPM is the active policy; the existing queue-length fallback to
FCFS does not silently inflate the counter.

Raw failing-before and passing-after outputs are retained under `raw/`. The
test uses the public `SchedulePolicy.calc_priority` path with deterministic
prefix lengths to reproduce continuous displacement without claiming a model
or production latency reproduction.
