"""The batch prefix is fetched in parallel, and one bad item stays one bad item.

Every batch item needs a download, an ffmpeg transcode and an ffprobe before
the GPU sees anything. Serially that prefix was most of a 32-item request's wall
clock. These stub the fetch and the model, so what is measured is the shape of
the loop, not the network or the weights.

    python tests/test_parallel_fetch.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asr_core  # noqa: E402

FETCH_SECONDS = 0.2
ITEMS = 16


def install_stubs(bad_indices=()):
    """Replace the fetch and the model, and count how many fetches overlap."""
    state = {"in_flight": 0, "peak": 0}
    lock = threading.Lock()

    def fake_materialize(value, workdir, idx=0):
        with lock:
            state["in_flight"] += 1
            state["peak"] = max(state["peak"], state["in_flight"])
        try:
            time.sleep(FETCH_SECONDS)
            if idx in bad_indices:
                raise ConnectionResetError("[Errno 104] Connection reset by peer")
            return (f"{workdir}/audio_{idx}_16k.wav", 4.0, "url", 64_000)
        finally:
            with lock:
                state["in_flight"] -= 1

    asr_core._materialize_audio = fake_materialize
    asr_core._get_model = lambda: (None, None)
    asr_core._transcribe_batch = lambda paths, lang, prompt, mnt: [
        {"language": lang, "text": f"text for {os.path.basename(p)}"} for p in paths
    ]
    return state


def run(n, bad_indices=()):
    state = install_stubs(bad_indices)
    req = {"audio": [f"https://example.test/clip-{i}.wav" for i in range(n)],
           "language": "Italian"}
    t0 = time.perf_counter()
    out = asr_core.transcribe(req)
    return out, time.perf_counter() - t0, state


def test_fetches_overlap():
    out, elapsed, state = run(ITEMS)
    serial = ITEMS * FETCH_SECONDS
    assert state["peak"] > 1, "fetches never overlapped: the loop is still serial"
    assert state["peak"] <= asr_core.FETCH_WORKERS, (
        f"peak {state['peak']} exceeded FETCH_WORKERS {asr_core.FETCH_WORKERS}")
    assert elapsed < serial / 2, (
        f"{ITEMS} items took {elapsed:.2f}s; serial would be {serial:.2f}s")
    assert out["count"] == ITEMS and out["failed"] == 0
    print(f"       {ITEMS} items in {elapsed:.2f}s (serial would be {serial:.2f}s), "
          f"peak {state['peak']} concurrent fetches")


def test_one_failed_fetch_costs_only_that_item():
    out, _, _ = run(ITEMS, bad_indices={3, 11})
    assert out["count"] == ITEMS, "the batch must still answer for every clip"
    assert out["failed"] == 2, f"expected 2 failures, got {out['failed']}"
    results = out["results"]
    # The error must sit on the item that failed, at the index it was asked at.
    assert "Connection reset" in results[3]["error"]
    assert "Connection reset" in results[11]["error"]
    assert all("error" not in results[i] for i in range(ITEMS) if i not in {3, 11})


def test_results_stay_in_the_order_asked():
    out, _, _ = run(ITEMS)
    for i, row in enumerate(out["results"]):
        assert f"audio_{i}_16k.wav" in row["text"], (
            f"item {i} came back as {row['text']!r}: results were reordered")


def test_single_item_still_works():
    out, _, state = run(1)
    assert out["text"].endswith("audio_0_16k.wav")
    assert state["peak"] == 1


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
