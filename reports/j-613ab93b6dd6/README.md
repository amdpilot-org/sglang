# FastAPI included-router metrics investigation

The recorded base already guards matching FastAPI routes that lack a `path`
attribute. The regression demonstrates the exact reported pre-fix exception on
FastAPI's real `_IncludedRouter`, then verifies that the checked-out SGLang
helper and Prometheus middleware do not raise.

Environment: FastAPI 0.141.1, Starlette 1.6.0, prepared repository interpreter
at `/tmp/amdpilot-repo-j-613ab93b6dd6/venv/bin/python`.

Run:

```bash
/tmp/amdpilot-repo-j-613ab93b6dd6/venv/bin/python -m pytest -q \
  test/registered/unit/utils/test_common.py -k GetFastAPIRequestPath
/tmp/amdpilot-repo-j-613ab93b6dd6/venv/bin/python \
  reports/j-613ab93b6dd6/reproduce_http_middleware.py
```

The original multi-node Ascend/NPU model command was not run. It is unnecessary
to reach this middleware failure, and the required hardware and weights are not
available in the assigned single-GPU environment.
