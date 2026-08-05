# Qwen3-ASR — Serverless Speech Recognition

[![Runpod](https://api.runpod.io/badge/Varshneyhars/Qwen3-ASR)](https://console.runpod.io/hub/Varshneyhars/Qwen3-ASR)

Production-ready serverless deployment of [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) — Alibaba's open-source multilingual speech recognition model (1.7 B params, 30 languages + 22 Chinese dialects).

Call it like any HTTP API: send an audio URL or base64 clip, get back a transcript with the detected language, optional word timestamps, or subtitles.

**Two deployment targets, one codebase:**

| Target | Guide | Entrypoint |
| --- | --- | --- |
| **RunPod Serverless** | this README | [handler.py](handler.py) — queue worker |
| **Google Cloud Run (GPU)** | **[GCP.md](GCP.md)** | [server.py](server.py) — HTTP service |

Both share [asr_core.py](asr_core.py), so the request schema, output shape, and every `QWEN_*` setting behave identically on either platform.

## Features

- **52 languages and dialects** — 30 languages plus 22 Chinese dialects, with automatic language identification returned on every response
- **Context biasing via `prompt`** — feed domain vocabulary, names, or jargon and the model biases recognition toward them. This is the headline feature over most ASR models.
- **Forced language** — pin the language to skip auto-detection
- **Word + segment timestamps** via the companion [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- **Subtitle output** — SRT and WebVTT generated server-side from word timings
- Audio input via URL, base64, or `data:` URI — any format ffmpeg can decode
- Long audio handled by automatic windowing with overlap dedup
- Batch requests — one bad item doesn't sink the rest

## Repository Layout

| File | Purpose |
| --- | --- |
| [asr_core.py](asr_core.py) | **Shared core** — ffmpeg normalization, Qwen3-ASR + aligner, chunking, SRT/VTT. No platform SDK. |
| [handler.py](handler.py) | RunPod serverless entrypoint (queue worker) |
| [server.py](server.py) | Cloud Run entrypoint (FastAPI HTTP service) |
| [Dockerfile](Dockerfile) | RunPod image — PyTorch 2.8.0 + CUDA 12.8, optional weight prefetch |
| [Dockerfile.cloudrun](Dockerfile.cloudrun) | Cloud Run image — same core, weights always baked in |
| [requirements.txt](requirements.txt) | Shared core deps |
| [requirements-runpod.txt](requirements-runpod.txt) / [requirements-cloudrun.txt](requirements-cloudrun.txt) | Per-platform additions |
| [deploy/gcp/](deploy/gcp/) | Cloud Build config, deploy scripts (bash + PowerShell), service spec, smoke test |
| [test_input.json](test_input.json) | Sample payload for local testing |
| [client_example.py](client_example.py) | Python client for `/runsync` and `/run` endpoints |
| [API.md](API.md) | Full integration guide — all languages, error handling, cost |
| [GCP.md](GCP.md) | Google Cloud Run deployment guide |
| [.runpod/hub.json](.runpod/hub.json) | RunPod Hub listing: presets, configurable env vars |
| [.runpod/tests.json](.runpod/tests.json) | Hub CI tests run on every release |
| [.dockerignore](.dockerignore) / [.gcloudignore](.gcloudignore) / [.gitignore](.gitignore) | Build and VCS hygiene |

## Why the `-hf` variant

Qwen ships two packagings of this model. This repo uses **`Qwen/Qwen3-ASR-1.7B-hf`**:

| | `Qwen3-ASR-1.7B-hf` (used here) | `Qwen3-ASR-1.7B` |
| --- | --- | --- |
| Library | `transformers>=5.13` (native) | `qwen-asr` package (v0.0.6) |
| Context biasing | ✅ `prompt` parameter | ✅ |
| Timestamps | Separate aligner model | Built in |
| Maturity | Standard transformers stack | `0.0.x` — very early |

The `-hf` route avoids depending on a `0.0.x` package for a production endpoint. The cost is that word timestamps need the 1.8 GB aligner as a second model — baked into the image alongside the ASR weights, and only loaded when you actually request timestamps.

## Requirements

- **GPU**: NVIDIA with ≥ 12 GB VRAM (L4 / A5000 / RTX 4090 / A100). ~4 GB for the ASR model in bf16, plus ~2 GB if the aligner loads.
- **CUDA**: ≥ 12.0
- **Python**: ≥ 3.10 (handled by the Docker base image)
- **Disk**: ~12 GB container + ~6 GB for weights

## Quick Start

### 1. Build and push the image

```bash
docker build -t <your-registry>/qwen3-asr-runpod:1.0.0 .
docker push <your-registry>/qwen3-asr-runpod:1.0.0
```

By default both models are baked in (`PREFETCH_MODEL=true`) for fast cold starts. Options:

```bash
# Skip the 1.8 GB aligner (no timestamps, smaller image)
docker build --build-arg PREFETCH_ALIGNER=false -t <registry>/qwen3-asr-runpod:1.0.0 .

# Skip baking entirely, download onto a network volume at runtime
docker build --build-arg PREFETCH_MODEL=false -t <registry>/qwen3-asr-runpod:1.0.0 .

# Use the smaller 0.6B model
docker build --build-arg MODEL_ID=Qwen/Qwen3-ASR-0.6B-hf -t <registry>/qwen3-asr-runpod:1.0.0 .
```

**Always bump the tag** — RunPod caches images per tag on its workers, so re-pushing `:latest` leaves warm workers running old code.

### 2. Create the RunPod endpoint

1. RunPod Console → **Serverless** → **New Endpoint**
2. Source: **Docker Image** → paste `<your-registry>/qwen3-asr-runpod:1.0.0`
3. GPU filter: **L4** (cheapest that fits comfortably), or A5000 / 4090
4. Container disk: **20 GB**
5. **FlashBoot: ON** — biggest cold-start win, and it's free
6. **Idle timeout: 60–300 s** — keeps the process warm so consecutive requests skip the model load
7. Execution timeout: **≥ 600 s** if you send long audio
8. *(Optional)* attach a **Network Volume** at `/runpod-volume` when `PREFETCH_MODEL=false`

### 3. Call the endpoint

```bash
export RUNPOD_API_KEY=rpa_xxx
export RUNPOD_ENDPOINT_ID=xxxxxxxx
python client_example.py
```

Or with `curl`:

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": {"audio_url": "https://example.com/speech.mp3", "prompt": "Vocabulary: Kubernetes, Grafana, Prometheus."}}'
```

## API

### Request (`input` object)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `audio_url` | string | — | `http(s)` URL of the audio. |
| `audio_base64` | string | — | Base64 audio, padded or not. |
| `gcs_uri` | string | — | `gs://bucket/object`. Cloud Run only — read with the runtime service account. |
| `audio` | string \| array | — | Base64 string, or an array of items for batch mode. |
| `prompt` | string \| array | `null` | **Context biasing.** Free-form text or a word list to bias recognition. Aliases: `context`, `hotwords`. |
| `language` | string | `null` | Force the language (`"English"` or `"en"`). Omit to auto-detect. |
| `timestamps` | bool | `false` | Word + segment timings via the forced aligner. |
| `response_format` | string | `"json"` | `json`, `verbose_json`, `text`, `srt`, `vtt`. |
| `max_new_tokens` | int | `448` | Generation cap per audio window. |

Supply **exactly one** of `audio_url` / `audio_base64` / `gcs_uri` / `audio`. The `srt`, `vtt`, and `verbose_json` formats turn `timestamps` on automatically.

### Response

```json
{
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
```

On failure:

```json
{ "error": "ffmpeg could not decode the audio: ..." }
```

See [API.md](API.md) for the complete reference.

## Context Biasing

The reason to pick Qwen3-ASR over a pure acoustic model like Parakeet or Whisper. Pass domain vocabulary and the model biases toward it:

```json
{"input": {
  "audio_url": "https://example.com/standup.mp3",
  "prompt": "Vocabulary: Kubernetes, Grafana, Prometheus, Anirudh, Bengaluru."
}}
```

A list is converted to that form automatically:

```json
{"input": {"audio_url": "...", "prompt": ["Kubernetes", "Grafana", "Prometheus"]}}
```

The prompt is free-form — background context works too, not just word lists:

```json
{"input": {"audio_url": "...", "prompt": "A cardiology consultation discussing atrial fibrillation and anticoagulants."}}
```

Best for proper nouns, product names, technical jargon, and acronyms — the things ASR reliably gets wrong. It biases, it doesn't guarantee.

## Long Audio

Qwen3-ASR processes roughly **5 minutes per forward pass** — a hard model constraint. Past `QWEN_CHUNK_THRESHOLD_S` (default 280 s) the handler splits into 4-minute windows with 5 s overlap, transcribes sequentially, offsets timestamps, and stitches the results. Reported as `metrics.strategy: "chunked"`.

Overlap dedup drops any word starting inside the previous window's accepted span, and the final text is rebuilt from deduplicated words so overlap phrases aren't repeated.

Hard cap is 4 hours (`QWEN_MAX_AUDIO_DURATION_S`). Use `/run` + `/status` polling for long audio, never `/runsync`.

## Local Testing

```bash
pip install -r requirements-runpod.txt
python handler.py
```

The RunPod SDK auto-detects `test_input.json`, runs the handler once, and prints the result. To serve a local HTTP API instead:

```bash
python handler.py --rp_serve_api --rp_api_host 0.0.0.0
curl -X POST http://localhost:8000/runsync -H 'Content-Type: application/json' -d @test_input.json
```

To exercise the Cloud Run server locally instead (needs a CUDA GPU for real inference, but the routes and validation work on CPU):

```bash
pip install -r requirements-cloudrun.txt
python server.py                  # serves on :8080
curl localhost:8080/ready
curl -X POST localhost:8080/transcribe -H 'Content-Type: application/json' -d @test_input.json
```

## Deploy on Google Cloud

Cloud Run is GCP's only true serverless GPU — it scales to zero and bills per second. Full walkthrough in **[GCP.md](GCP.md)**; the short version:

```bash
gcloud config set project YOUR_PROJECT_ID
./deploy/gcp/deploy.sh              # or: .\deploy\gcp\deploy.ps1 -ProjectId YOUR_PROJECT_ID
./deploy/gcp/smoke_test.sh
```

Cloud Run GPUs exist in only six regions — `us-central1`, `us-east4`, `europe-west1`, `europe-west4`, `asia-southeast1`, and `asia-south1` (Mumbai, invitation only). Projects get 3 GPUs of quota on first deploy, so no quota request is normally needed. See [GCP.md § Prerequisites](GCP.md#prerequisites).

The Cloud Run service adds three endpoints on top of the shared schema: `POST /transcribe` (native), `POST /v1/audio/transcriptions` (OpenAI-compatible file upload), and `POST /runsync` (RunPod-shaped envelope, so an existing client only changes its base URL). Cloud Run caps requests at 60 minutes — see [GCP.md § The 60-minute ceiling](GCP.md#the-60-minute-ceiling) for long audio.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `QWEN_MODEL_ID` | `Qwen/Qwen3-ASR-1.7B-hf` | ASR model |
| `QWEN_ALIGNER_MODEL_ID` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | Timestamp model |
| `QWEN_ENABLE_ALIGNER` | `true` | Set false to reject timestamp requests and never load the aligner |
| `QWEN_PRECISION` | `bfloat16` | `bfloat16`, `float16`, or `float32` |
| `QWEN_MAX_NEW_TOKENS` | `448` | Generation cap per window |
| `QWEN_CHUNK_THRESHOLD_S` | `280` | Switch to windowed chunking |
| `QWEN_CHUNK_LENGTH_S` | `240` | Window length (model limit is ~5 min) |
| `QWEN_CHUNK_OVERLAP_S` | `5` | Window overlap |
| `QWEN_MAX_AUDIO_DURATION_S` | `14400` | Reject longer audio (4 h) |
| `QWEN_MAX_DOWNLOAD_BYTES` | `536870912` | Reject larger payloads (512 MB) |
| `QWEN_DOWNLOAD_TIMEOUT_S` | `120` | URL fetch timeout |
| `QWEN_MAX_BATCH_ITEMS` | `32` | Max audio items per request |
| `QWEN_BLOCK_PRIVATE_URLS` | `true` | SSRF guard on `audio_url` |
| `QWEN_SEGMENT_MAX_GAP_S` | `0.6` | Pause that breaks a subtitle cue |
| `QWEN_SEGMENT_MAX_CHARS` | `84` | Max characters per cue |
| `QWEN_SEGMENT_MAX_DURATION_S` | `6.0` | Max seconds per cue |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` | `/opt/hf-cache` (prefetched) | Where weights live |

`bfloat16` is the model's native dtype and the right default. It falls back to `float32` automatically on CPU.

## Performance

| Metric | Value |
| --- | --- |
| Parameters | 1.7 B (+ 0.6 B aligner) |
| VRAM | ~4 GB ASR, ~6 GB with aligner loaded |
| Languages | 30 + 22 Chinese dialects |
| Max audio per pass | ~5 minutes (chunked automatically beyond that) |
| dtype | bfloat16 |

Qwen3-ASR is autoregressive, so it's slower than CTC/TDT models like Parakeet — expect single-digit to low-tens realtime factors rather than hundreds. You trade raw throughput for context biasing, language ID, and dialect coverage. Check `metrics.realtime_factor` on your own audio.

## Cost Tips

- Bake weights into the image (`PREFETCH_MODEL=true`) — no HF fetch on worker boot. Biggest cold-start win.
- **Turn FlashBoot on** and set **idle timeout to 60–300 s**. Both are free, and a short idle timeout means every request re-pays the model load.
- Set `QWEN_ENABLE_ALIGNER=false` and build with `--build-arg PREFETCH_ALIGNER=false` if you never need timestamps — saves 1.8 GB of image and ~2 GB of VRAM.
- Send URLs rather than base64 when you can; base64 inflates the payload ~33%.
- Use the 0.6 B model (`Qwen/Qwen3-ASR-0.6B-hf`) if accuracy allows — roughly a third the weights.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Slow first request | Image didn't prefetch — rebuild with `PREFETCH_MODEL=true`, or keep a min-worker warm. |
| Model load repeats every request | Idle timeout too short. Raise it to 60–300 s. |
| `ffmpeg could not decode the audio` | URL returned HTML/JSON, or corrupt upload. Verify with `curl -I`. |
| Timestamps missing from response | Set `"timestamps": true`, or check `QWEN_ENABLE_ALIGNER` isn't `false`. |
| Timestamps silently empty | Alignment failed and was skipped so you'd still get the transcript — check worker logs for `[align] failed`. |
| `CUDA out of memory` | Aligner + ASR need ~6 GB. Use a ≥ 12 GB GPU, or set `QWEN_ENABLE_ALIGNER=false`. |
| Transcript truncated | Raise `max_new_tokens`. |
| Prompt seems ignored | Biasing is probabilistic, not a guarantee. Keep prompts short and specific. |
| Wrong language detected | Pass `"language": "English"` to force it. |
| `Refusing to fetch ...` | SSRF guard. Set `QWEN_BLOCK_PRIVATE_URLS=false` if intentional. |

## License & Usage

- **Code (this repo)**: MIT
- **Model (Qwen3-ASR weights)**: Apache-2.0 — commercial use permitted
- **Ethical use**: Don't transcribe people without consent where the law requires it.

## Links

- Model: [huggingface.co/Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- Forced aligner: [huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- Smaller variant: [huggingface.co/Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- Upstream code: [github.com/QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- RunPod Serverless: [docs.runpod.io/serverless](https://docs.runpod.io/serverless)
