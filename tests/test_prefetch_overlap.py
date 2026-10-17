"""One request's audio downloads while another request owns the GPU.

This is the behaviour that was missing. The GPU limit used to wrap the whole of
asr_core.transcribe, so a queued request could not even begin downloading until
the request ahead of it had finished generating: the device waited out the next
request's entire fetch prefix, every request.

The GPU is now gated around the forward pass alone, and the fetch pool is shared
across the process, so the two overlap. These tests drive the real
_transcribe_batch (including its _gpu_slot) against a fake torch and a fake
model, so what is measured is the gating, not the weights.

    python tests/test_prefetch_overlap.py
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- a torch and a model small enough to run on a laptop --------------------
class _NoOpCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_fake_torch = types.ModuleType("torch")
_fake_torch.inference_mode = _NoOpCtx
sys.modules.setdefault("torch", _fake_torch)

import asr_core  # noqa: E402

FETCH_S = 0.4
GPU_S = 0.4
ITEMS = 8

spans = []            # (tag, phase, start, end)
spans_lock = threading.Lock()
gpu_live = {"now": 0, "peak": 0}


def record(tag, phase, start):
    with spans_lock:
        spans.append((tag, phase, start, time.perf_counter()))


class _Ids:
    """Stands in for the token tensors: only .shape and 2-D slicing are used."""
    def __init__(self, n):
        self.n = n
        self.shape = (1, n)

    def __getitem__(self, key):
        return self


class _Inputs(dict):
    def to(self, *a, **k):
        return self


class _Processor:
    def apply_transcription_request(self, **kw):
        self._n = len(kw["audio"]) if isinstance(kw.get("audio"), list) else 1
        return _Inputs(input_ids=_Ids(self._n))

    def decode(self, generated, return_format=None):
        return [{"language": "Italian", "transcription": f"clip {i}"}
                for i in range(self._n)]


class _Model:
    device = "cpu"
    dtype = "float32"

    def generate(self, **kw):
        t0 = time.perf_counter()
        with spans_lock:
            gpu_live["now"] += 1
            gpu_live["peak"] = max(gpu_live["peak"], gpu_live["now"])
        try:
            time.sleep(GPU_S)
        finally:
            with spans_lock:
                gpu_live["now"] -= 1
        record(threading.current_thread().name, "gpu", t0)
        return _Ids(1)


def install_stubs():
    spans.clear()
    gpu_live.update(now=0, peak=0)
    proc = _Processor()
    model = _Model()
    asr_core._get_model = lambda: (model, proc)

    def fake_materialize(value, workdir, idx=0):
        t0 = time.perf_counter()
        time.sleep(FETCH_S / ITEMS * min(ITEMS, asr_core.FETCH_WORKERS))
        record(threading.current_thread().name, "fetch", t0)
        return (f"{workdir}/audio_{idx}_16k.wav", 4.0, "url", 64_000)

    asr_core._materialize_audio = fake_materialize


def one_request(tag):
    threading.current_thread().name = tag
    return asr_core.transcribe({
        "audio": [f"https://example.test/{tag}-{i}.wav" for i in range(ITEMS)],
        "language": "Italian",
    })


def overlaps(a, b):
    return a[2] < b[3] and b[2] < a[3]


def test_b_fetches_while_a_holds_the_gpu():
    install_stubs()
    out = {}
    threads = [threading.Thread(target=lambda t=t: out.update({t: one_request(t)}), name=t)
               for t in ("A", "B")]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - t0

    assert out["A"]["count"] == ITEMS and out["B"]["count"] == ITEMS, out

    gpu_spans = [s for s in spans if s[1] == "gpu"]
    fetch_spans = [s for s in spans if s[1] == "fetch"]
    assert gpu_spans, "no GPU work was recorded"

    # The point of the change: some request fetched during another's GPU pass.
    crossed = [(f, g) for g in gpu_spans for f in fetch_spans
               if f[0] != g[0] and overlaps(f, g)]
    assert crossed, (
        "no fetch overlapped another request's GPU pass: the gate is still "
        "wrapped around the fetch")

    # And the GPU itself stayed serialised, which is what GPU_SLOTS=1 means.
    assert gpu_live["peak"] == asr_core.GPU_SLOTS, (
        f"{gpu_live['peak']} concurrent forward passes, GPU_SLOTS is "
        f"{asr_core.GPU_SLOTS}")

    fully_serial = 2 * (FETCH_S + GPU_S)
    assert elapsed < fully_serial * 0.9, (
        f"two requests took {elapsed:.2f}s; serialising everything would be "
        f"{fully_serial:.2f}s, so nothing overlapped")
    print(f"       2 requests in {elapsed:.2f}s (no overlap would be "
          f"{fully_serial:.2f}s), {len(crossed)} fetch/GPU overlaps, "
          f"peak {gpu_live['peak']} concurrent pass(es)")


def test_the_gate_is_actually_around_the_forward_pass():
    # Structural, so a refactor that widens the gate back over the fetch fails
    # here rather than quietly costing throughput in production.
    import inspect
    src = inspect.getsource(asr_core._transcribe_batch)
    assert "_gpu_slot()" in src, "_transcribe_batch no longer takes a GPU slot"
    grouped = inspect.getsource(asr_core._transcribe_grouped)
    assert "_gpu_slot" not in grouped, (
        "_transcribe_grouped holds the GPU slot itself, which would cover the "
        "fetch loop again")
    assert "_get_fetch_pool()" in grouped, "the fetch pool is no longer shared"


def test_fetch_pool_is_one_per_process():
    assert asr_core._get_fetch_pool() is asr_core._get_fetch_pool()
    assert asr_core._get_fetch_pool()._max_workers == asr_core.FETCH_WORKERS


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:
            failures += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
