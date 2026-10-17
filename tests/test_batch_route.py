"""The /batch contract, with the model stubbed out.

/batch is what the annotation platform's ingest calls, and what it needs from
every response is an id it did not have to infer from position. These cover the
mapping, not the model: asr_core.transcribe is replaced, so they run on any
machine with no GPU and no weights.

    python -m pytest tests/test_batch_route.py      (or: python tests/test_batch_route.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QWEN_PRELOAD_MODEL", "false")
os.environ.setdefault("QWEN_WARMUP_INFERENCE", "false")

from fastapi.testclient import TestClient  # noqa: E402

import asr_core  # noqa: E402
import server  # noqa: E402

client = TestClient(server.app)
_calls = []


def stub(reply):
    """Point asr_core.transcribe at a canned reply and record what it was asked."""
    def fake(payload):
        _calls.append(payload)
        return reply(payload) if callable(reply) else reply
    asr_core.transcribe = fake
    _calls.clear()


def ok_batch(payload):
    """What asr_core returns for a healthy multi-item batch."""
    return {
        "results": [
            {"text": f"clip {i}", "language": payload.get("language"), "duration": 4.0}
            for i in range(len(payload["audio"]))
        ],
        "count": len(payload["audio"]),
        "failed": 0,
    }


def items(n):
    return [{"id": f"id-{i}", "gcs_uri": f"https://example.test/clip-{i}.wav"} for i in range(n)]


def test_32_items_round_trip():
    stub(ok_batch)
    r = client.post("/batch", json={"items": items(32), "language": "Italian"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 32
    assert body["errors"] == []
    assert [row["id"] for row in body["results"]] == [f"id-{i}" for i in range(32)]
    assert [row["text"] for row in body["results"]] == [f"clip {i}" for i in range(32)]
    # The caller's language must reach the model, not be replaced by a default.
    assert _calls[0]["language"] == "Italian"
    assert len(_calls[0]["audio"]) == 32


def test_language_defaults_to_english():
    stub(ok_batch)
    r = client.post("/batch", json={"items": items(3)})
    assert r.status_code == 200, r.text
    # The prototype hardcoded English; callers that send nothing must keep it.
    assert _calls[0]["language"] == "English"
    assert all(row["language"] == "English" for row in r.json()["results"])


def test_single_item_is_still_a_list():
    # asr_core flattens a one-item batch into the single-item shape.
    stub({"text": "only one", "language": "English", "duration": 2.0})
    r = client.post("/batch", json={"items": items(1)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 1
    assert body["results"][0] == {"id": "id-0", "text": "only one",
                                  "language": "English", "duration": 2.0}


def test_per_item_failure_keeps_its_id_and_spares_the_rest():
    stub({
        "results": [
            {"text": "fine", "language": "English"},
            {"error": "ConnectionResetError: [Errno 104] Connection reset by peer", "index": 1},
            {"text": "also fine", "language": "English"},
        ],
        "count": 3,
        "failed": 1,
    })
    r = client.post("/batch", json={"items": items(3)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [row["id"] for row in body["results"]] == ["id-0", "id-2"]
    assert body["errors"] == [{"id": "id-1", "error":
                               "ConnectionResetError: [Errno 104] Connection reset by peer"}]
    assert body["n"] == 2


def test_short_result_list_does_not_lose_clips():
    # A clip the model never answered for must come back as an error, not a gap.
    stub({"results": [{"text": "one", "language": "English"}], "count": 1, "failed": 0})
    r = client.post("/batch", json={"items": items(3)})
    body = r.json()
    assert [e["id"] for e in body["errors"]] == ["id-1", "id-2"]


def test_whole_request_failure_is_400():
    stub({"error": "Batch of 40 exceeds the limit of 32."})
    r = client.post("/batch", json={"items": items(3)})
    assert r.status_code == 400
    assert "exceeds the limit" in r.json()["error"]


def test_bad_bodies_are_400():
    stub(ok_batch)
    assert client.post("/batch", json={}).status_code == 400
    assert client.post("/batch", json={"items": []}).status_code == 400
    assert client.post("/batch", json={"items": [{"id": "x"}]}).status_code == 400
    assert client.post("/batch", json={"items": items(1), "language": "   "}).status_code == 400


def test_sources_alias_and_bare_strings():
    stub(ok_batch)
    r = client.post("/batch", json={"sources": ["https://example.test/a.wav"]})
    assert r.status_code == 200, r.text
    # No id sent, so the source stands in for one rather than an index.
    assert r.json()["results"][0]["id"] == "https://example.test/a.wav"


def test_conflicting_per_item_context_is_refused():
    stub(ok_batch)
    both = [{"id": "a", "gcs_uri": "https://e.test/a.wav", "context": "alpha"},
            {"id": "b", "gcs_uri": "https://e.test/b.wav", "context": "beta"}]
    assert client.post("/batch", json={"items": both}).status_code == 400
    same = [dict(it, context="shared") for it in both]
    r = client.post("/batch", json={"items": same})
    assert r.status_code == 200, r.text
    assert _calls[0]["prompt"] == "shared"


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
