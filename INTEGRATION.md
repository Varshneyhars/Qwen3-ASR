# Qwen3-ASR API — Integration Guide

Speech-to-text service running on Google Cloud Run. Send audio, get back a transcript with detected language, optional word-level timestamps, and subtitles.

---

## Endpoint

| | |
| --- | --- |
| **Base URL** | `https://qwen3-asr-qdzv24yrua-uc.a.run.app` |
| **Region / Project** | `us-central1` / `foundary-gcp` |
| **Auth** | Google IAM — OIDC identity token required (see below) |
| **Model** | Qwen3-ASR-1.7B + Qwen3-ForcedAligner-0.6B |
| **Hardware** | 1× NVIDIA L4 GPU, 8 vCPU, 32 GiB |
| **Max request duration** | 3600s (Cloud Run hard limit) |
| **Concurrency** | 1 request per instance, autoscales 0 → 3 |

The service scales to zero when idle, so **the first request after a quiet period takes 60–90 seconds** while a GPU instance starts. Subsequent requests are fast. Build retries and generous timeouts around this — it is expected behavior, not a fault.

---

## Authentication

Cloud Run needs an **OIDC identity token** — *not* an OAuth access token. The `audience` must be the service's base URL. This is the single most common integration mistake.

### Credentials

A service account has already been provisioned for application use:

```text
asr-client@foundary-gcp.iam.gserviceaccount.com   (roles/run.invoker)
```

The project owner will supply a JSON key file for it **through a private channel** — password manager, encrypted message, or your secret store. It is not in this repository and must never be committed: **this repo is public.**

Once you have it:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/secure/path/asr-client.json
```

The Google auth libraries pick that up automatically. In production, load it from your platform's secret manager rather than a file on disk.

**If your backend already runs on GCP** (Cloud Run, GKE, Compute Engine, App Engine), skip the key file entirely — attach the `asr-client` service account to the workload and the libraries resolve credentials from the metadata server. Nothing secret ever exists on disk. Prefer this whenever possible.

### Tokens

Identity tokens are valid for **1 hour**. The client libraries below cache and refresh automatically. Do not mint one per request — it adds latency and burns quota.

---

## Website integration

> **Critical: call this from your backend, never from browser JavaScript.**
>
> The service account key grants the ability to spend GPU time on your project. Anything shipped to the browser — bundled JS, a public env var, a fetch from the client — is readable by every visitor. There is also no CORS configuration on this service, so direct browser calls will be blocked regardless.

The correct shape is a thin proxy on your own server:

```text
Browser ──upload──> Your backend ──ID token──> Qwen3-ASR (Cloud Run)
   ▲                     │
   └──── transcript ─────┘
```

Your backend holds the credential, enforces your own auth/rate limits, and forwards the audio.

### Node.js / Express backend

```js
const express = require('express');
const multer = require('multer');
const { GoogleAuth } = require('google-auth-library');

const ASR_URL = 'https://qwen3-asr-qdzv24yrua-uc.a.run.app';
const auth = new GoogleAuth();
const upload = multer({ limits: { fileSize: 100 * 1024 * 1024 } });
const app = express();

app.post('/api/transcribe', upload.single('audio'), async (req, res) => {
  try {
    // Enforce YOUR app's auth here before spending GPU time.
    if (!req.user) return res.status(401).json({ error: 'unauthorized' });

    const client = await auth.getIdTokenClient(ASR_URL);   // token cached internally
    const { data } = await client.request({
      url: `${ASR_URL}/transcribe`,
      method: 'POST',
      data: {
        audio_base64: req.file.buffer.toString('base64'),
        response_format: 'verbose_json',
        language: 'English',
      },
      timeout: 600000,   // cold start can take 90s — do NOT use a 30s default
    });
    res.json({ text: data.text, words: data.words });
  } catch (e) {
    const status = e.response?.status || 500;
    res.status(status).json({ error: e.response?.data?.error || 'transcription failed' });
  }
});

app.listen(3000);
```

### Python / FastAPI backend

```python
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
import google.auth.transport.requests, google.oauth2.id_token
import requests

ASR_URL = "https://qwen3-asr-qdzv24yrua-uc.a.run.app"
app = FastAPI()
_session = requests.Session()

def _token() -> str:
    # fetch_id_token caches internally; safe to call per request.
    return google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), ASR_URL)

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    raw = await audio.read()
    resp = _session.post(
        f"{ASR_URL}/transcribe",
        headers={"Authorization": f"Bearer {_token()}"},
        json={"audio_base64": base64.b64encode(raw).decode(),
              "response_format": "verbose_json"},
        timeout=600,   # cold start tolerance
    )
    if resp.status_code != 200:
        raise HTTPException(resp.status_code, resp.json().get("error", "failed"))
    d = resp.json()
    return {"text": d["text"], "words": d.get("words", [])}
```

### Browser side

Just post to *your* endpoint — no Google credentials involved:

```js
const fd = new FormData();
fd.append('audio', fileInput.files[0]);
const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
const { text } = await res.json();
```

### For large files, skip base64

Base64 inflates payloads ~33%. For anything above a few MB, have the browser upload straight to Cloud Storage with a signed URL, then pass the `gs://` URI:

```json
{ "gcs_uri": "gs://your-bucket/uploads/abc123.webm", "response_format": "verbose_json" }
```

The bucket must grant `roles/storage.objectViewer` to `qwen3-asr-sa@foundary-gcp.iam.gserviceaccount.com`.

### Things that will bite you

| Issue | Handling |
| --- | --- |
| **Cold start** | First request after idle takes 60–90s. Set client timeouts ≥ 120s. Show a progress state; don't fail at 30s. |
| **Long audio** | Anything past ~45 min exceeds the 3600s cap. Queue those as background jobs rather than blocking an HTTP response. |
| **Concurrency** | Only 3 requests process at once. Beyond that they queue. Add a job queue if your traffic is bursty. |
| **Retries** | Retry `429`/`503`/network with backoff. Never retry `400` — it re-bills for the same rejection. |
| **User-facing latency** | Even warm, a 5-minute recording takes ~50s. Use a job + polling pattern rather than a synchronous request for anything long. |

---

## Quick start

```bash
SERVICE_URL=https://qwen3-asr-qdzv24yrua-uc.a.run.app
TOKEN=$(gcloud auth print-identity-token)

curl -X POST "$SERVICE_URL/transcribe" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/speech.mp3", "response_format": "verbose_json"}'
```

---

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/transcribe` | **Primary endpoint.** JSON in, JSON out. |
| `POST` | `/v1/audio/transcriptions` | OpenAI-compatible multipart file upload. |
| `POST` | `/runsync` | Same as `/transcribe`, RunPod-style response envelope. |
| `GET` | `/health` | Liveness. Never blocks. |
| `GET` | `/ready` | 503 until the model is loaded on the GPU. |
| `POST` | `/warmup` | Blocks until loaded. Use to pre-warm before a burst. |
| `GET` | `/docs` | Interactive OpenAPI documentation. |

---

## `POST /transcribe`

### Request fields

Supply **exactly one** audio source:

| Field | Type | Description |
| --- | --- | --- |
| `audio_url` | string | Public `http(s)` URL. Signed URLs work. |
| `gcs_uri` | string | `gs://bucket/object`. Read with the service's own identity — the bucket must grant `roles/storage.objectViewer` to `qwen3-asr-sa@foundary-gcp.iam.gserviceaccount.com`. |
| `audio_base64` | string | Base64 audio, padding optional. |
| `audio` | string \| array | Base64 string, **or an array for batch mode** (see below). |

Optional:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `language` | string | auto | Force language, e.g. `"English"`, `"Chinese"`. Omit to auto-detect — the detected value is returned either way. |
| `prompt` | string \| array | — | **Context biasing.** Domain vocabulary, names, jargon. An array is joined automatically. Aliases: `context`, `hotwords`. |
| `timestamps` | bool | `false` | Word + segment timings. Forced on for `srt`/`vtt`/`verbose_json`. |
| `response_format` | string | `"json"` | `json`, `verbose_json`, `text`, `srt`, `vtt` |
| `max_new_tokens` | int | `448` | Raise if dense speech comes back truncated. |

Any audio format ffmpeg can decode is accepted — mp3, m4a, wav, ogg, opus, webm, and video containers. Everything is normalized to 16 kHz mono internally.

### Response

```json
{
  "text": "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
  "language": "English",
  "segments": [
    { "start": 0.56, "end": 5.44, "text": "Mr Quilter is the apostle of the middle classes..." }
  ],
  "words": [
    { "start": 0.56, "end": 0.80, "text": "Mr" },
    { "start": 0.80, "end": 1.28, "text": "Quilter" }
  ],
  "duration": 5.855,
  "metrics": {
    "audio_seconds": 5.855,
    "fetch_seconds": 0.772,
    "inference_seconds": 1.167,
    "realtime_factor": 5.0,
    "strategy": "single_pass",
    "source_kind": "url",
    "input_bytes": 187404,
    "prompt_used": false,
    "language_forced": false
  },
  "model": { "model_id": "Qwen/Qwen3-ASR-1.7B-hf", "device": "cuda", "gpu": "NVIDIA L4" },
  "total_seconds": 1.939
}
```

`segments` and `words` appear only when timestamps are requested. With `response_format: "json"` they are nested under `timestamps.word` / `timestamps.segment` instead; `verbose_json` puts them at the top level as shown.

`srt` and `vtt` add an `srt` / `vtt` string field containing the subtitle document.

---

## Batch mode

**Use this for volume.** Batching amortizes the cold start and connection setup across many files. For short clips it is roughly **20× cheaper** than one request per file.

```json
{
  "audio": [
    { "gcs_uri": "gs://my-bucket/clip001.wav" },
    { "gcs_uri": "gs://my-bucket/clip002.wav" }
  ],
  "response_format": "text",
  "language": "English"
}
```

Response:

```json
{
  "results": [
    { "text": "...", "language": "English", "metrics": { } },
    { "error": "ffmpeg could not decode the audio: ...", "index": 1 }
  ],
  "count": 2,
  "failed": 1
}
```

**Maximum 32 items per request.** Results are returned in the same order as the inputs. One bad item does not fail the batch — check each entry for an `error` key.

Guidance:

- Keep batches at 32. Larger gains nothing (per-file cost is linear) and increases how much you must retry on failure.
- Keep total audio per batch under ~30 minutes to stay well inside the 3600s timeout.
- Send batches back-to-back so the instance stays warm.

A ready-made bulk client with retries and resume support is in `batch_client.py`.

---

## Code examples

### Python

```python
import google.auth.transport.requests
import google.oauth2.id_token
import requests

SERVICE_URL = "https://qwen3-asr-qdzv24yrua-uc.a.run.app"

def get_token() -> str:
    """OIDC identity token. Audience MUST be the service base URL."""
    return google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), SERVICE_URL
    )

def transcribe(audio_url: str, **opts) -> dict:
    resp = requests.post(
        f"{SERVICE_URL}/transcribe",
        headers={"Authorization": f"Bearer {get_token()}"},
        json={"audio_url": audio_url, "response_format": "verbose_json", **opts},
        timeout=3600,   # cold start can take 90s; long audio far more
    )
    resp.raise_for_status()
    return resp.json()

result = transcribe("https://example.com/speech.mp3",
                    prompt="Vocabulary: Kubernetes, Grafana, Prometheus.")
print(result["text"])
```

Requires `pip install google-auth requests`. Cache the token for an hour rather than calling `get_token()` per request.

### Node.js

```js
const { GoogleAuth } = require('google-auth-library');

const SERVICE_URL = 'https://qwen3-asr-qdzv24yrua-uc.a.run.app';
const auth = new GoogleAuth();

async function transcribe(audioUrl, opts = {}) {
  // getIdTokenClient handles minting and refreshing the OIDC token.
  const client = await auth.getIdTokenClient(SERVICE_URL);
  const res = await client.request({
    url: `${SERVICE_URL}/transcribe`,
    method: 'POST',
    data: { audio_url: audioUrl, response_format: 'verbose_json', ...opts },
    timeout: 3600000,
  });
  return res.data;
}

transcribe('https://example.com/speech.mp3').then(r => console.log(r.text));
```

Requires `npm install google-auth-library`.

### File upload (OpenAI-compatible)

```bash
curl -X POST "$SERVICE_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@meeting.m4a \
  -F response_format=srt
```

Works with the official OpenAI SDKs by overriding the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url=f"{SERVICE_URL}/v1", api_key=get_token())
with open("meeting.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="qwen3-asr", file=f).text)
```

`response_format` of `text`, `srt`, and `vtt` return a bare body; `json` returns the full object.

---

## Errors

| Status | Meaning | Action |
| --- | --- | --- |
| `400` | Bad input — missing/multiple audio sources, undecodable audio, unknown `response_format`. Body: `{"error": "..."}` | Fix the request. Do not retry. |
| `401` | Missing or malformed token. | Check the `Authorization` header. |
| `403` | Caller lacks `roles/run.invoker`, or the token expired (1 hour). | Grant the role, or refresh the token. |
| `413` | Upload over 512 MB. | Split the audio. |
| `429` | Too many concurrent requests. | Back off and retry. |
| `503` | All inference slots busy, or model still loading. | Retry with backoff. |
| `504` | Request exceeded 3600s. | Split the audio into shorter pieces. |

Retry `429`, `503`, and network errors with exponential backoff. Never retry `400` — it will fail identically and still costs GPU time.

---

## Limits

| Limit | Value | Notes |
| --- | --- | --- |
| Max audio duration | 4 hours | But see the timeout below — practical ceiling is lower. |
| Max request duration | 3600s | Cloud Run hard limit. ~45 min of audio at observed speed. |
| Max upload / download | 512 MB | Per audio file. |
| Batch items | 32 | Per request. |
| Concurrent requests | 3 | Equals max instances. Excess queues. |

Audio longer than ~4.5 minutes is automatically split into overlapping windows and stitched back together — transparent to the caller, reported as `metrics.strategy: "chunked"`.

---

## Performance

Measured on this deployment:

| Workload | Throughput |
| --- | --- |
| Single file, warm | **5–9× realtime** (60s audio ≈ 8.6s) |
| Batch of 32 × 20s clips | **6.1× realtime** (640s audio ≈ 105s) |
| Cold start | 60–90s before the first response |

Word timestamps add roughly 0.1s per minute of audio — effectively free.

The model is autoregressive, so it is slower than CTC models like Whisper-tiny or Parakeet. The trade is context biasing, language identification, and 52 languages/dialects.

---

## Capabilities worth knowing

**Context biasing** is the main reason to use this model. Pass domain vocabulary and recognition is biased toward it:

```json
{ "audio_url": "...", "prompt": "Vocabulary: Kubernetes, Grafana, Anirudh, Bengaluru." }
```

Free-form context also works: `"A cardiology consultation discussing atrial fibrillation."` Most effective on proper nouns, product names, and acronyms. It biases; it does not guarantee.

**Languages:** 30 languages plus 22 Chinese dialects, auto-identified. Pass `language` to skip detection when you already know it.

**Subtitles:** `srt` and `vtt` are generated server-side from word timings, with cue breaks on sentence boundaries, pauses, and length limits.

---

## Operational notes

- **Cold starts are normal.** Scale-to-zero means no cost when idle, at the price of a 60–90s first request. Call `POST /warmup` ahead of a known burst.
- **Timeouts:** set the client timeout to at least 120s for short audio; longer for big files. A default 30s HTTP timeout will fail on every cold start.
- **Idempotency:** the API has no request de-duplication. A retry re-transcribes and re-bills.
- **Logs:** every request emits a structured line with `request_id`, `seconds`, and `audio_seconds`. Send `X-Request-Id` and it is echoed back and logged for correlation.

## Contacts

- Service owner: `foundary-gcp` project admins
- Runtime service account: `qwen3-asr-sa@foundary-gcp.iam.gserviceaccount.com`
- Current revision: `qwen3-asr-00001-dnw`
- Source: <https://github.com/Varshneyhars/Qwen3-ASR>
