# ReplaySSM extra-buffer investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/37834

Mirror issue: https://github.com/amdpilot-org/sglang/issues/912

The recorded base still forced ReplaySSM onto `no_buffer` and did not reset
`replayssm_write_pos` when recycled slots entered the extra-buffer ping-pong
pool. The regression fixture failed before the source correction with stale
cursors `[5, 7]` on allocation and `9` on donation replacement.

The correction resets cursors at both ownership boundaries. It admits
`extra_buffer` for GDN, whose decode path force-flushes at radix tracking
boundaries, while retaining the guard for KDA, where that force-flush is absent.

Raw command output is retained beside this report. The original Qwen3.8-27B on
H200 TTFT reproduction was not possible with the assigned gfx950 GPU and no
model weights. The GPU evidence here is limited to the actual device-tensor
cursor reset and does not stand in for the unavailable model/architecture run.
