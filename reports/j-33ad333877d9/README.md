# Abort cleanup investigation

Upstream issue: https://github.com/sgl-project/sglang/issues/34113

Mirror issue: https://github.com/amdpilot-org/sglang/issues/1669

The current HTTP `/abort_request` handler directly dispatches an `AbortReq`; it
does not call `create_abort_task`. A live tiny-Llama request on the assigned
gfx950 returned HTTP 200 and then ended with `finish_reason.type == "abort"`,
with no `AttributeError` in the server log.

The reported exception was nevertheless reproduced in the adjacent response
cleanup path. `StreamingResponse` installs `create_abort_task` before its body
iterator starts. If the response cleanup runs before `generate_request` has
normalized the request, `GenerateReqInput.is_single` does not exist. The
focused regression in `evidence/unit/failing-before.txt` records that exact
failure. Cleanup now determines single versus batch from the public `rid`
shape, which exists when the request object is constructed.

## Verification

- `python -m pytest -q test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py -k TestCreateAbortTask`
  passes the unstarted-stream regression and active single/batch boundaries.
- The complete tokenizer-manager cleanup test file passes.
- `HIP_VISIBLE_DEVICES=0 python reports/j-33ad333877d9/run_http_abort.py ...`
  launched a real server with the qualified deterministic tiny Llama fixture,
  generated tokens on the MI355X, returned HTTP 200 from `/abort_request`, and
  produced an abort finish reason after four generated tokens.

## Limitations

The original `meta-llama/Llama-3.2-1B-Instruct` weights and NVIDIA RTX 4090
environment were not available. The tiny random Llama fixture validates HTTP
transport and real engine execution only; it does not validate semantic model
quality or the original CUDA platform. The owned server process did not exit
within the runner's 30-second shutdown grace period and was killed and reaped
after the successful probe. No native code was changed or rebuilt.
