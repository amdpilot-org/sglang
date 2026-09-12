# Grammar compile executor investigation

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`, the prepared process
reported 384 CPUs and `BaseGrammarBackend` created the stdlib-default 32-worker
executor. The installed xgrammar API reports a default `max_threads=8`, matching
the nested-concurrency concern in the issue.

The correction caps automatic outer concurrency at 8, uses half the visible CPU
count below that cap, and provides `SGLANG_GRAMMAR_COMPILE_MAX_WORKERS` for an
explicit positive override (`0` selects automatic sizing). Raw commands and
outputs are retained in `raw/`.

This environment's cgroup v1 CPU controller reports no CFS quota, so it does not
validate throttling or decode-latency impact. The assigned gfx950 device was
inventoried but was not used: executor sizing is a CPU-only path and no model
serving or numerical GPU claim is relevant.
