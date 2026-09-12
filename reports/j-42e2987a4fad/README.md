# Diffusion startup profiler validation

The raw Qwen-Image/MI355X run is retained in `qwen_image_gpu.log`. It used the
exact `Qwen/Qwen-Image` revision
`75e0b4be04f60ec59a75f475837eced720f823b6`, downloaded to the private runtime
directory outside the checkout.

The emitted startup tree shows the requested hierarchy from Scheduler setup to
individual pipeline components. The generated image remains outside the git
worktree at the path recorded in the log; its SHA256 is
`4fb0c7ceedb9da5d54393af7e1033e9ef9364c514d28ccb72d208a166a51a7c3`.

See `result.json` for commands, measured claims, and unverified paths.
