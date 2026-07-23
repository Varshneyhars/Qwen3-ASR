# Qwen3-ASR RunPod API Reference

Complete integration guide for the Qwen3-ASR serverless endpoint. Use this to call the API from any language or framework.

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Request body](#request-body)
- [Response body](#response-body)
- [Context biasing](#context-biasing)
- [Language handling](#language-handling)
- [Response formats](#response-formats)
- [Audio input formats](#audio-input-formats)
- [Batch requests](#batch-requests)
- [Long audio](#long-audio)
- [Code examples](#code-examples)
  - [cURL](#curl)
  - [Python](#python)
  - [Node.js](#nodejs)
  - [JavaScript (browser)](#javascript-browser)
  - [Go](#go)
- [Error handling](#error-handling)
- [Timing and cost](#timing-and-cost)
- [Best practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

| Property | Value |
| --- | --- |
| Base URL | `https://api.runpod.ai/v2/{ENDPOINT_ID}` |
| Auth scheme | Bearer token in `Authorization` header |
| Content-Type | `application/json` |
| Model | `Qwen/Qwen3-ASR-1.7B-hf` |
| Aligner | `Qwen/Qwen3-ForcedAligner-0.6B-hf` (timestamps only) |
| Languages | 30 + 22 Chinese dialects, auto-detected |
| Input sample rate | Anything — normalized to 16 kHz mono server-side by ffmpeg |

---

## Authentication

```http
Authorization: Bearer rpa_XXXXXXXXXXXXXXXXXXXXX
Content-Type: application/json
```

Get the key from [RunPod Console → Settings → API Keys](https://runpod.io/console/user/settings).

> **Security**: Never embed the key in client-side code. Put a backend proxy in front and let your frontend call that.

---

## Endpoints

| Method | Path | Purpose | Response |
| --- | --- | --- | --- |
| `POST` | `/run` | Submit asynchronously. Returns a job ID immediately. | `{ "id": "...", "status": "IN_QUEUE" }` |
| `POST` | `/runsync` | Submit and block until complete (or 30 s). | Full result, or `{ "status": "IN_QUEUE", "id": "..." }` |
| `GET` | `/status/{id}` | Poll an async job. | `{ "status": "...", "output": {...}, "executionTime": ms, "delayTime": ms }` |
| `POST` | `/cancel/{id}` | Cancel a queued or running job. | `{ "status": "CANCELLED" }` |
| `GET` | `/health` | Endpoint health + worker counts. | `{ "jobs": {...}, "workers": {...} }` |

### Recommended usage patterns

- **Short clips (< 2 min), warm endpoint**: `/runsync` with a 60–120 s client timeout.
- **Long audio, cold endpoint, or production**: `/run` + poll `/status/{id}` every 2–3 s. **Required** past ~5 minutes of audio.
- **Never** poll faster than every 1 s.

---

## Request body

```json
{
  "input": {
    "audio_url": "string (one of three sources)",
    "audio_base64": "string (one of three sources)",
    "audio": "string | array (one of three sources)",
    "prompt": "Vocabulary: Quilter, apostle, gospel.",
    "language": "English",
    "timestamps": false,
    "response_format": "json",
    "max_new_tokens": 448
  }
}
```

### Field reference

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `audio_url` | string | — | `http(s)` URL. Must be publicly reachable (no auth headers sent). |
| `audio_base64` | string | — | Base64-encoded audio bytes. Padding optional. |
| `audio` | string \| array | — | Raw base64 string, or an array for [batch mode](#batch-requests). |
| `prompt` | string \| array | `null` | [Context biasing](#context-biasing). Aliases: `context`, `hotwords`. |
| `language` | string | `null` | Force the language. Omit to auto-detect. |
| `timestamps` | bool | `false` | Word + segment timings. Loads the aligner on first use. |
| `response_format` | string | `"json"` | `json`, `verbose_json`, `text`, `srt`, `vtt`. |
| `max_new_tokens` | int | `448` | Generation cap per audio window. Raise for dense speech. |

Supply **exactly one** audio source. Supplying two returns a validation error rather than silently picking one.

`srt`, `vtt`, and `verbose_json` set `timestamps: true` automatically.

---

## Response body

### Successful transcription

```json
{
  "delayTime": 112,
  "executionTime": 934,
  "id": "df4ca305-5b8b-4c64-8042-04ca268dec40-e2",
  "status": "COMPLETED",
  "output": {
    "text": "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
    "language": "English",
    "timestamps": {
      "word": [{"start": 0.56, "end": 0.80, "text": "Mr"}],
      "segment": [{"start": 0.56, "end": 6.20, "text": "Mr. Quilter is the apostle..."}]
    },
    "duration": 7.12,
    "metrics": {
      "audio_seconds": 7.12,
      "fetch_seconds": 0.31,
      "inference_seconds": 0.44,
      "realtime_factor": 16.2,
      "strategy": "single_pass",
      "source_kind": "url",
      "input_bytes": 228044,
      "prompt_used": true,
      "language_forced": false
    },
    "model": {
      "model_id": "Qwen/Qwen3-ASR-1.7B-hf",
      "device": "cuda",
      "dtype": "torch.bfloat16",
      "gpu": "NVIDIA L4",
      "load_seconds": 24.1,
      "aligner_enabled": true
    },
    "total_seconds": 0.79
  }
}
```

| Field | Description |
| --- | --- |
| `output.text` | The transcript. Always present. |
| `output.language` | Detected (or forced) language. Always present. |
| `output.timestamps.word` | Word timings from the aligner. |
| `output.timestamps.segment` | Subtitle-sized cues grouped from words. |
| `output.srt` / `output.vtt` | Present for those formats. |
| `output.metrics.strategy` | `single_pass` or `chunked`. |
| `output.metrics.prompt_used` | Whether context biasing was applied. |
| `output.metrics.language_forced` | Whether you pinned the language. |
| `delayTime` | ms in queue + container init. |
| `executionTime` | ms of handler time. |

### Status values

| Value | Meaning |
| --- | --- |
| `IN_QUEUE` | Waiting for a worker. |
| `IN_PROGRESS` | Worker running your handler. |
| `COMPLETED` | Finished; `output` populated (may contain `output.error`). |
| `FAILED` | Handler raised, or platform gave up. |
| `CANCELLED` | Cancelled by caller. |
| `TIMED_OUT` | Exceeded the endpoint's max execution time. |

---

## Context biasing

The main reason to choose Qwen3-ASR. Pass domain vocabulary and the model biases recognition toward it — the practical fix for proper nouns, product names, jargon, and acronyms.

### As a sentence

```json
{"input": {
  "audio_url": "https://example.com/standup.mp3",
  "prompt": "Vocabulary: Kubernetes, Grafana, Prometheus, Anirudh, Bengaluru."
}}
```

### As a list

Converted to `Vocabulary: a, b, c.` automatically:

```json
{"input": {"audio_url": "...", "prompt": ["Kubernetes", "Grafana", "Prometheus"]}}
```

### As free-form context

The prompt isn't restricted to word lists — background context works too:

```json
{"input": {
  "audio_url": "...",
  "prompt": "A cardiology consultation discussing atrial fibrillation, warfarin dosing, and anticoagulant therapy."
}}
```

### Notes

- **It biases, it doesn't guarantee.** A word in the prompt can still be missed.
- **Keep it short and specific.** Dumping hundreds of words dilutes the effect and eats context.
- The aliases `context` and `hotwords` are accepted and behave identically.
- `metrics.prompt_used` confirms it was applied.

---

## Language handling

**Auto-detect (default)** — omit `language`. The detected value comes back in `output.language` either way.

```json
{"input": {"audio_url": "https://example.com/speech.mp3"}}
```

**Forced** — pass a name or code. Useful for short/noisy clips where detection is unreliable, or code-switched audio you want pinned.

```json
{"input": {"audio_url": "https://example.com/speech.mp3", "language": "Chinese"}}
```

Both `"Chinese"` and `"zh"` work. `metrics.language_forced` tells you which path ran.

---

## Response formats

| Format | Returns |
| --- | --- |
| `json` (default) | `text`, `language`, plus `timestamps` if requested |
| `verbose_json` | `text`, `language`, flat `segments` / `words` arrays, `duration` |
| `text` | `text` + `language` only — smallest payload |
| `srt` | `text`, `language`, `srt` string |
| `vtt` | `text`, `language`, `vtt` string |

`verbose_json` is closest to the OpenAI Whisper API shape, so it's the easiest drop-in if you're migrating.

Subtitle cues are grouped from word timings, breaking on sentence punctuation, pauses over 0.6 s, 84 characters, or 6 seconds — all tunable via `QWEN_SEGMENT_*` env vars.

---

## Audio input formats

### 1. HTTPS URL

```json
"audio_url": "https://cdn.example.com/interview.mp3"
```

Publicly reachable, 120 s timeout, 512 MB cap. URLs resolving to private, loopback, link-local, or reserved IPs are **rejected** (SSRF guard).

### 2. Raw base64

```json
"audio_base64": "UklGRiQAAABXQVZFZm10..."
```

No prefix needed, padding optional.

### 3. Data URI

```json
"audio": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10..."
```

The `data:...,` prefix is stripped automatically. Must be base64-encoded.

### Container and codec support

Everything is normalized through ffmpeg to 16 kHz mono 16-bit PCM, so anything ffmpeg decodes works:

**Audio**: wav, mp3, flac, m4a/aac, ogg, opus, wma, aiff, amr
**Video**: mp4, mov, mkv, webm, avi — first audio track extracted, video discarded

Multi-channel is downmixed to mono; any sample rate is resampled.

---

## Batch requests

```json
{"input": {
  "audio": [
    {"audio_url": "https://example.com/a.mp3"},
    {"audio_url": "https://example.com/b.wav"}
  ],
  "prompt": "Vocabulary: Kubernetes, Grafana.",
  "timestamps": true
}}
```

Response:

```json
{
  "results": [
    {"text": "First clip.", "language": "English", "metrics": {...}},
    {"error": "Downloaded 0 bytes from https://example.com/b.wav", "index": 1}
  ],
  "count": 2,
  "failed": 1,
  "model": {...},
  "total_seconds": 3.11
}
```

**One item's failure does not fail the job.** Check `failed` and inspect each result for an `error` key; failed items carry their `index`.

`prompt` and `language` are request-level — they apply to every item in the batch. For different vocabulary per clip, send separate requests.

Capped at 32 items (`QWEN_MAX_BATCH_ITEMS`).

---

## Long audio

Qwen3-ASR processes roughly **5 minutes per forward pass** — a hard model constraint, not a tuning choice.

| Duration | `metrics.strategy` | Mechanism |
| --- | --- | --- |
| ≤ 280 s | `single_pass` | One forward pass |
| > 280 s | `chunked` | 4-min windows, 5 s overlap, timestamps offset and stitched |

Overlap dedup drops any word starting inside the previous window's accepted span, and text is rebuilt from deduplicated words so overlap phrases aren't repeated.

Hard cap is 4 hours. **Always use `/run` + `/status` for long audio** — a 1-hour file takes minutes of wall clock, and `/runsync` will just hand you a job ID to poll anyway.

---

## Code examples

### cURL

```bash
# Basic
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/speech.mp3"}}' \
  | jq -r '.output.text'

# With context biasing
curl -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/standup.mp3",
                 "prompt": "Vocabulary: Kubernetes, Grafana, Prometheus."}}' \
  | jq -r '.output | "[\(.language)] \(.text)"'

# Subtitles to a file
curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/talk.mp4", "response_format": "srt"}}' \
  | jq -r '.output.srt' > talk.srt

# Async submit + poll (use for long audio)
JOB=$(curl -s -X POST "https://api.runpod.ai/v2/$ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/2h-podcast.mp3", "response_format": "srt"}}' \
  | jq -r '.id')

while true; do
  R=$(curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
    "https://api.runpod.ai/v2/$ENDPOINT_ID/status/$JOB")
  S=$(echo "$R" | jq -r '.status'); echo "$S"
  [[ "$S" == "COMPLETED" ]] && echo "$R" | jq -r '.output.srt' > podcast.srt && break
  [[ "$S" == "FAILED" || "$S" == "CANCELLED" ]] && echo "$R" && exit 1
  sleep 3
done
```

### Python

```python
import base64
import os
import time

import requests

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def transcribe(source, *, is_file=False, **kwargs) -> dict:
    payload = {"input": dict(kwargs)}
    if is_file:
        with open(source, "rb") as f:
            payload["input"]["audio_base64"] = base64.b64encode(f.read()).decode()
    else:
        payload["input"]["audio_url"] = source

    r = requests.post(f"{BASE}/run", json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    job_id = r.json()["id"]

    while True:
        s = requests.get(f"{BASE}/status/{job_id}", headers=HEADERS, timeout=30).json()
        status = s["status"]
        if status == "COMPLETED":
            out = s["output"]
            if "error" in out:
                raise RuntimeError(out["error"])
            return out
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"{status}: {(s.get('output') or {}).get('error', s.get('error'))}")
        time.sleep(2)


if __name__ == "__main__":
    out = transcribe(
        "https://example.com/standup.mp3",
        prompt=["Kubernetes", "Grafana", "Prometheus"],
        timestamps=True,
        response_format="verbose_json",
    )
    print(f"[{out['language']}] {out['text']}")
    for w in out["words"][:10]:
        print(f"{w['start']:6.2f}-{w['end']:6.2f}  {w['text']}")

    srt = transcribe("talk.mp4", is_file=True, response_format="srt")["srt"]
    open("talk.srt", "w", encoding="utf-8").write(srt)
```

### Node.js

```javascript
import fs from "node:fs";

const API_KEY = process.env.RUNPOD_API_KEY;
const ENDPOINT_ID = process.env.RUNPOD_ENDPOINT_ID;
const BASE = `https://api.runpod.ai/v2/${ENDPOINT_ID}`;
const HEADERS = {
  Authorization: `Bearer ${API_KEY}`,
  "Content-Type": "application/json",
};

async function transcribe(source, { isFile = false, ...opts } = {}) {
  const input = { ...opts };
  if (isFile) {
    input.audio_base64 = fs.readFileSync(source).toString("base64");
  } else {
    input.audio_url = source;
  }

  const submit = await fetch(`${BASE}/run`, {
    method: "POST",
    headers: HEADERS,
    body: JSON.stringify({ input }),
  });
  const { id } = await submit.json();

  while (true) {
    await new Promise((r) => setTimeout(r, 2000));
    const s = await (await fetch(`${BASE}/status/${id}`, { headers: HEADERS })).json();
    if (s.status === "COMPLETED") {
      if (s.output.error) throw new Error(s.output.error);
      return s.output;
    }
    if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(s.status)) {
      throw new Error(`${s.status}: ${s.output?.error ?? s.error}`);
    }
  }
}

const out = await transcribe("https://example.com/speech.mp3", {
  prompt: "Vocabulary: Kubernetes, Grafana.",
});
console.log(`[${out.language}] ${out.text}`);
```

### JavaScript (browser)

```javascript
async function transcribeFile(file, opts = {}) {
  const base = `https://api.runpod.ai/v2/${ENDPOINT_ID}`;
  const headers = { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json" };

  // FileReader gives a data URI; the handler accepts it as-is.
  const dataUri = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

  const { id } = await fetch(`${base}/run`, {
    method: "POST",
    headers,
    body: JSON.stringify({ input: { audio: dataUri, ...opts } }),
  }).then((r) => r.json());

  while (true) {
    await new Promise((r) => setTimeout(r, 2500));
    const s = await fetch(`${base}/status/${id}`, { headers }).then((r) => r.json());
    if (s.status === "COMPLETED") {
      if (s.output.error) throw new Error(s.output.error);
      return s.output;
    }
    if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(s.status)) {
      throw new Error(s.output?.error ?? s.error);
    }
  }
}
```

> ⚠️ Put a backend proxy between your frontend and RunPod. Exposing your API key in client JS lets anyone drain your credits.

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "time"
)

func do(method, url string, body any) (map[string]any, error) {
    var buf io.Reader
    if body != nil {
        b, _ := json.Marshal(body)
        buf = bytes.NewReader(b)
    }
    req, _ := http.NewRequest(method, url, buf)
    req.Header.Set("Authorization", "Bearer "+os.Getenv("RUNPOD_API_KEY"))
    req.Header.Set("Content-Type", "application/json")
    res, err := http.DefaultClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer res.Body.Close()
    out := map[string]any{}
    _ = json.NewDecoder(res.Body).Decode(&out)
    return out, nil
}

func main() {
    base := "https://api.runpod.ai/v2/" + os.Getenv("RUNPOD_ENDPOINT_ID")
    sub, _ := do("POST", base+"/run", map[string]any{
        "input": map[string]any{
            "audio_url": "https://example.com/speech.mp3",
            "prompt":    "Vocabulary: Kubernetes, Grafana.",
        },
    })
    id := sub["id"].(string)

    for {
        time.Sleep(2 * time.Second)
        s, _ := do("GET", base+"/status/"+id, nil)
        status := s["status"].(string)
        if status == "COMPLETED" {
            out := s["output"].(map[string]any)
            if e, ok := out["error"]; ok {
                fmt.Println("error:", e)
                return
            }
            fmt.Printf("[%v] %v\n", out["language"], out["text"])
            return
        }
        if status == "FAILED" || status == "CANCELLED" || status == "TIMED_OUT" {
            fmt.Println("error:", s)
            return
        }
    }
}
```

---

## Error handling

### Categories

| Where | HTTP | Shape | Retryable? |
| --- | --- | --- | --- |
| Network / DNS | n/a | exception | Yes, with backoff |
| Auth | 401 | `{"error":"Unauthorized"}` | No — fix key |
| RunPod platform | 500 | `{"error":"..."}` | Yes, after 5–10 s |
| Handler validation | 200 + `output.error` | `"input: provide one of [...]"` | No — fix payload |
| Audio fetch/decode | 200 + `output.error` | `"ffmpeg could not decode..."` | Sometimes — check the URL |
| Handler runtime | 200 + `status: FAILED` | `{"output":{"error":"..."}}` | Sometimes (OOM, CUDA) |
| Job timeout | 200 + `status: TIMED_OUT` | `{"error":"job timed out..."}` | Yes, with async or shorter audio |

### Common handler errors

| Message | Cause |
| --- | --- |
| `input: provide one of ['audio', 'audio_url', ...]` | No audio source in the payload. |
| `input: got multiple audio sources [...]` | Sent both `audio_url` and `audio_base64`. |
| `Input is not valid base64: ...` | Truncated or corrupted base64. |
| `Refusing to fetch 'x': resolves to non-public address` | SSRF guard tripped. |
| `ffmpeg could not decode the audio: ...` | Not a media file, or corrupt container. |
| `Audio is Ns, over the 14400s limit.` | Longer than 4 h; split client-side. |
| `Timestamps require the forced aligner, which is disabled` | `QWEN_ENABLE_ALIGNER=false` on this endpoint. |
| `'language' must be a non-empty string` | Passed an empty string or non-string. |
| `'prompt' must be a string or a list of strings` | Passed a dict or number. |
| `Batch of N exceeds the limit of 32.` | Too many items. |

### Reading errors

```python
s = response.json()
if s.get("status") == "COMPLETED":
    out = s["output"]
    if "error" in out:           # handler-level failure
        raise RuntimeError(out["error"])
    text = out["text"]
else:
    err = (s.get("output") or {}).get("error") or s.get("error") or f"Unknown: {s}"
    raise RuntimeError(err)
```

### A note on silent timestamp failures

If forced alignment fails, the handler **logs and returns the transcript without timestamps** rather than failing the whole request — losing alignment shouldn't cost you the transcription. So an empty `words` array with a healthy `text` means alignment was skipped. Check worker logs for `[align] failed`.

---

## Timing and cost

### Latency profile

| Scenario | Wall time | Notes |
| --- | --- | --- |
| First request (spin-up + model load) | 40–90 s | Once per worker; weights baked into the image |
| Warm, 10 s clip | 1–3 s | |
| Warm, 10 s clip + timestamps | 2–5 s | Aligner load on first use, then ~1 s extra |
| Warm, 5 min file | 15–45 s | Single pass |
| Warm, 1 h file | 4–12 min | Chunked; async required |

Qwen3-ASR is autoregressive, so it's substantially slower than CTC/TDT models — expect single-digit to low-tens realtime factors. You trade throughput for context biasing, language ID, and dialect coverage. Measure `metrics.realtime_factor` on your own audio.

### Cost levers

- **Idle timeout**: the big one. Too short and every request re-pays the ~25 s model load. 60–300 s is the sweet spot.
- **FlashBoot**: free, cuts container boot dramatically. Turn it on.
- **Min workers**: eliminates cold starts entirely but bills 24/7.
- **GPU choice**: L4 ≈ $0.00019/s is the right pick — the model needs ~6 GB with the aligner, so a bigger card buys nothing.
- **Disable the aligner** if you never need timestamps: `QWEN_ENABLE_ALIGNER=false` plus `--build-arg PREFETCH_ALIGNER=false` saves 1.8 GB image and ~2 GB VRAM.

---

## Best practices

1. **Use a backend proxy.** Never ship RunPod keys to end-user clients.

2. **Set idle timeout to 60–300 s.** The single highest-impact setting. A 5 s timeout means every request reloads the model.

3. **Use `/run` + `/status` past ~2 minutes of audio.** `/runsync` caps at 30 s regardless.

4. **Send URLs, not base64, when you can.** Base64 inflates payloads ~33% and traverses the API twice.

5. **Keep prompts short and specific.** 5–20 domain terms beats a 500-word dump — long prompts dilute biasing and eat context.

6. **Force the language for short clips.** Auto-detection needs a few seconds of clean speech; under ~3 s it can misfire.

7. **Only request timestamps when you need them.** They trigger a second model and an extra forward pass.

8. **Raise the endpoint execution timeout** before sending long audio, or you'll get `TIMED_OUT` with no partial result.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `401 Unauthorized` | Bad / rotated key | Generate a new one |
| `IN_QUEUE` forever | No workers available | Raise Max Workers |
| Every request shows a model load | Idle timeout too short | Raise to 60–300 s |
| `FAILED: job timed out` | Cold worker or long audio | Min Workers ≥ 1, raise execution timeout, use `/run` |
| `CUDA out of memory` | ASR + aligner ≈ 6 GB | Use ≥ 12 GB GPU or `QWEN_ENABLE_ALIGNER=false` |
| `ffmpeg could not decode the audio` | URL served HTML, or corrupt upload | `curl -I` the URL |
| Empty `words`, healthy `text` | Alignment failed, skipped by design | Check logs for `[align] failed` |
| Transcript cut off mid-sentence | Hit `max_new_tokens` | Raise it |
| Prompt appears ignored | Biasing is probabilistic | Shorten and target the prompt |
| Wrong language | Short or noisy clip | Pass `language` explicitly |
| Repeated words at chunk edges | Overlap dedup gap | Raise `QWEN_CHUNK_OVERLAP_S` |
| Slow (`realtime_factor` < 2) | Running on CPU | Check `model.device` in the response |

---

## Reference resources

- Model card: [huggingface.co/Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- Forced aligner: [huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- Upstream code: [github.com/QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- RunPod serverless docs: [docs.runpod.io/serverless](https://docs.runpod.io/serverless)
- This endpoint's handler: [handler.py](handler.py)
- In-repo examples: [client_example.py](client_example.py), [test_input.json](test_input.json)
