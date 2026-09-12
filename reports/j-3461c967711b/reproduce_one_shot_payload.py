"""Reproduce the one-shot FD property behind upstream issue 35498."""

import multiprocessing as mp

import torch

from sglang.srt.utils import MultiprocessingSerializer


def consume(index, payload, start, results):
    start.wait()
    try:
        restored = MultiprocessingSerializer.deserialize(payload)
        results.put((index, "ok", restored["w"].numel()))
    except BaseException as exc:
        results.put((index, type(exc).__name__, str(exc)))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    tensors = {"w": torch.zeros(1024 * 1024, dtype=torch.bfloat16)}
    payload = MultiprocessingSerializer.serialize(tensors, output_str=True)
    start = ctx.Event()
    results = ctx.Queue()
    workers = [
        ctx.Process(target=consume, args=(index, payload, start, results))
        for index in range(8)
    ]
    for worker in workers:
        worker.start()
    start.set()
    output = sorted(results.get(timeout=30) for _ in workers)
    for worker in workers:
        worker.join()
    print(output)
