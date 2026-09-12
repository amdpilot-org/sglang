# Issue 36105 investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/36105

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1250

At base commit `358c163250ad3b1f62939b01ce1314a0a31a0365`, `serve()` built a
`ServeRequest` with `try_get_model_path(dispatch_argv)`. That helper only
scanned explicit `--model-path` and `--model` arguments. The subsequent
required-model check therefore raised before `_run_llm()` could call
`prepare_server_args()`, where `ConfigArgumentMerger` normally reads YAML.

The baseline traceback in `raw/baseline-serve-config.log` reproduces that exact
ordering with the prepared repository implementation. After the correction,
`raw/fixed-serve-config.log` shows the YAML model path reaching the
`run_server(ServerArgs)` launch boundary. Only `run_server` was replaced with a
capture function in this post-fix check, avoiding a model download while still
exercising backend selection and the real server-argument parser. The
nonexistent model is intentional and no inference or GPU validation is claimed.

Two upstream pull requests addressing the same issue were reviewed before the
change: https://github.com/sgl-project/sglang/pull/36107 and
https://github.com/sgl-project/sglang/pull/36168. Both were open and unmerged.
This implementation follows the narrower approach of reusing the repository's
real `ConfigArgumentMerger`, avoiding a second YAML/key-precedence parser.
