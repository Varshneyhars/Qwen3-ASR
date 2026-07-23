"""Example client for calling the deployed Qwen3-ASR RunPod endpoint.

Usage:
    export RUNPOD_API_KEY=...
    export RUNPOD_ENDPOINT_ID=...
    python client_example.py
"""
import base64
import os
import sys
import time

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def run_sync(payload: dict, timeout: int = 300) -> dict:
    r = requests.post(f"{BASE}/runsync", json=payload, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run_async(payload: dict, poll_interval: float = 2.0) -> dict:
    r = requests.post(f"{BASE}/run", json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    job_id = r.json()["id"]
    while True:
        s = requests.get(f"{BASE}/status/{job_id}", headers=HEADERS, timeout=30).json()
        if s["status"] in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            return s
        time.sleep(poll_interval)


def encode_file(path: str) -> str:
    """Read a local audio file as base64 for the `audio_base64` field."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def show(result: dict, save_to: str | None = None) -> dict:
    out = result.get("output") or {}
    if "error" in out:
        print(f"[error] {out['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"[{out.get('language')}] {out.get('text')}")
    m = out.get("metrics") or {}
    if m:
        print(f"  ({m['audio_seconds']:.1f}s audio in {m['inference_seconds']:.2f}s "
              f"= {m['realtime_factor']}x realtime, {m['strategy']}"
              f"{', prompt' if m.get('prompt_used') else ''})")

    if save_to:
        text = out.get("srt") or out.get("vtt") or out.get("text", "")
        with open(save_to, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"saved: {save_to}")
    return out


if __name__ == "__main__":
    AUDIO = ("https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/"
             "librispeech_mr_quilter.wav")

    # 1. Plain transcription, language auto-detected
    print("=== basic ===")
    show(run_sync({"input": {"audio_url": AUDIO}}))

    # 2. Context biasing — the headline Qwen3-ASR feature. Feeding domain
    #    vocabulary makes rare proper nouns come out right.
    print("\n=== with prompt (context biasing) ===")
    show(run_sync({"input": {
        "audio_url": AUDIO,
        "prompt": "Vocabulary: Quilter, apostle, gospel.",
    }}))

    #    A list is converted to "Vocabulary: a, b, c." automatically.
    print("\n=== prompt as a list ===")
    show(run_sync({"input": {
        "audio_url": AUDIO,
        "prompt": ["Quilter", "apostle", "gospel"],
    }}))

    # 3. Forced language (skips auto-detection)
    print("\n=== forced language ===")
    show(run_sync({"input": {"audio_url": AUDIO, "language": "English"}}))

    # 4. Word + segment timestamps via the forced aligner
    print("\n=== timestamps ===")
    out = show(run_sync({"input": {
        "audio_url": AUDIO,
        "timestamps": True,
        "response_format": "verbose_json",
    }}))
    for w in (out.get("words") or [])[:10]:
        print(f"   {w['start']:6.2f}-{w['end']:6.2f}  {w['text']}")

    # 5. Subtitles
    print("\n=== srt ===")
    show(run_sync({"input": {"audio_url": AUDIO, "response_format": "srt"}}), save_to="output.srt")

    # 6. Batch — one failure does not sink the others
    print("\n=== batch ===")
    batch = run_sync({"input": {"audio": [{"audio_url": AUDIO}, {"audio_url": AUDIO}]}})
    result = batch.get("output") or {}
    print(f"count={result.get('count')} failed={result.get('failed')}")
    for i, r in enumerate(result.get("results") or []):
        print(f"  [{i}] {r.get('error') or r.get('text', '')[:60]}")

    # 7. Long audio — use async so a cold start can't hit an HTTP timeout
    # show(run_async({"input": {"audio_url": "https://example.com/2h-podcast.mp3",
    #                           "response_format": "srt"}}), save_to="podcast.srt")

    # 8. Local file upload
    # show(run_sync({"input": {"audio_base64": encode_file("meeting.m4a"),
    #                          "response_format": "vtt"}}), save_to="meeting.vtt")
