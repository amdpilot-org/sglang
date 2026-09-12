from types import SimpleNamespace

from sglang.srt.managers.io_struct import (
    BatchTokenizedEmbeddingReqInput,
    BatchTokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import stamp_http_worker_ipc


def snapshot(batch_cls, *, rids=None, child_rids=("rid-0", "rid-1")):
    req = batch_cls(
        rids=rids,
        batch=[
            SimpleNamespace(rid=rid, http_worker_ipc=None) for rid in child_rids
        ],
    )
    stamp_http_worker_ipc(req, "ipc://review-worker")
    return {
        "type": batch_cls.__name__,
        "rids": req.rids,
        "http_worker_ipcs": req.http_worker_ipcs,
        "child_http_worker_ipcs": [item.http_worker_ipc for item in req.batch],
    }


def main():
    expected = {
        "rids": ["rid-0", "rid-1"],
        "http_worker_ipcs": ["ipc://review-worker", "ipc://review-worker"],
        "child_http_worker_ipcs": [
            "ipc://review-worker",
            "ipc://review-worker",
        ],
    }
    failed = False
    for cls in (BatchTokenizedGenerateReqInput, BatchTokenizedEmbeddingReqInput):
        got = snapshot(cls)
        print(got)
        for key, value in expected.items():
            if got[key] != value:
                failed = True
                print(f"MISMATCH {cls.__name__}.{key}: {got[key]!r} != {value!r}")

    empty = snapshot(BatchTokenizedGenerateReqInput, child_rids=())
    print(empty)
    if empty["rids"] != [] or empty["http_worker_ipcs"] != []:
        failed = True
        print("MISMATCH empty batch metadata")

    explicit = snapshot(
        BatchTokenizedGenerateReqInput,
        rids=["explicit-0", "explicit-1"],
    )
    print(explicit)
    if explicit["rids"] != ["explicit-0", "explicit-1"]:
        failed = True
        print("MISMATCH explicit rids were not preserved")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
