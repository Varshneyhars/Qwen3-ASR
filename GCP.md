# Deploying Qwen3-ASR on Google Cloud

Serverless GPU deployment on **Cloud Run**, with scale-to-zero and per-second billing.

The same `asr_core.py` powers both this and the RunPod worker — only the transport layer differs. Deploying here does not affect the RunPod endpoint.

---

## Why Cloud Run

GCP has three ways to run a GPU model. Only one is actually serverless:

| Option | Scales to zero | Billing | Verdict |
| --- | --- | --- | --- |
| **Cloud Run + GPU** | ✅ | Per second, only while an instance exists | **Use this** |
| Vertex AI Endpoints | ❌ (min 1 node) | Per node-hour, always on | Good for steady high traffic, not serverless |
| GKE Autopilot | ❌ | Per pod-hour | You manage a cluster |

Cloud Run attaches one **NVIDIA L4 (24 GB)** per instance. Qwen3-ASR needs ~4 GB, plus ~2 GB with the aligner loaded — comfortable headroom.

---

## Architecture

```text
Client ──HTTPS──> Cloud Run (L4 GPU, scale 0..N)
                    │  server.py     FastAPI, one request per instance
                    │  asr_core.py   ffmpeg → Qwen3-ASR → aligner → SRT/VTT
                    └─ weights baked into the image (no network volume exists)
                            │
                            └─ optional: gs:// audio via the runtime service account
```

The key difference from RunPod: **there is no job queue.** RunPod holds a request in a queue and hands it to a free worker. Cloud Run routes HTTP straight at a container and the connection stays open for the whole transcription. That drives three design choices in this deployment:

1. **`--concurrency 1`** — one GPU, one model, no batching. A second concurrent request would contend for CUDA memory with no throughput gain.
2. **Weights baked into the image** — Cloud Run instances have no persistent disk, so anything not in the image layer is re-downloaded on every cold start.
3. **Model preloaded during startup** — Cloud Run grants a CPU boost while an instance starts, and that time isn't charged to a user's request latency.

---

## Prerequisites

- `gcloud` CLI, authenticated: `gcloud auth login`
- A project with billing enabled
- **GPU quota.** Projects get a default allocation of **3 GPUs on first deployment**, so a single-instance deploy normally works without any request. You only need to ask when scaling past it:

  Console → **IAM & Admin → Quotas** → filter for
  `Total Nvidia L4 GPU allocation without zonal redundancy, per project per region`
  under the **Cloud Run Admin API**.

  Requests for the *non-redundant* quota are granted faster than the zonally-redundant one — which is one reason this deployment passes `--no-gpu-zonal-redundancy`.

- **A region that offers Cloud Run GPUs.** Only six do:

  | Region | Location | Notes |
  | --- | --- | --- |
  | `us-central1` | Iowa | **Default here.** Largest capacity. |
  | `us-east4` | N. Virginia | Good US East alternative. |
  | `europe-west1` | Belgium | |
  | `europe-west4` | Netherlands | |
  | `asia-southeast1` | Singapore | Closest generally-available region to India. |
  | `asia-south1` | Mumbai | ⚠️ **Invitation only** — request access via your GCP account team. |

  Scaling hard in `us-central1` or `us-east4` may need a quota increase. Check [Cloud Run GPU docs](https://cloud.google.com/run/docs/configuring/services/gpu) for the current list — it changes.

---

## Quick start

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Linux / macOS / Git Bash
./deploy/gcp/deploy.sh
```

```powershell
# Windows PowerShell
.\deploy\gcp\deploy.ps1 -ProjectId YOUR_PROJECT_ID
```

The script enables the APIs, creates the Artifact Registry repo and a least-privilege runtime service account, builds the image with Cloud Build, and deploys it with a GPU attached.

**Budget 20–35 minutes for the first build** — it downloads ~5 GB of weights and pushes a ~25 GB image. Redeploys are seconds:

```bash
SKIP_BUILD=1 ./deploy/gcp/deploy.sh
```

Then verify:

```bash
./deploy/gcp/smoke_test.sh
```

It checks liveness, waits out the cold start, transcribes a sample clip, verifies SRT cue timings, and confirms bad input is rejected with a 400.

---

## Calling the service

The service is deployed **private** by default — callers need `roles/run.invoker`.

```bash
SERVICE_URL=$(gcloud run services describe qwen3-asr --region us-central1 --format 'value(status.url)')
TOKEN=$(gcloud auth print-identity-token)
```

### Native endpoint

```bash
curl -X POST "$SERVICE_URL/transcribe" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "audio_url": "https://example.com/speech.mp3",
        "prompt": "Vocabulary: Kubernetes, Grafana, Prometheus.",
        "response_format": "verbose_json"
      }'
```

The request body is the same schema as the RunPod `input` object — see [API.md](API.md). A `{"input": {...}}` wrapper is accepted and unwrapped, so RunPod payloads work verbatim.

### Audio straight from Cloud Storage

Skips the download hop entirely and avoids base64 inflation for large files:

```bash
curl -X POST "$SERVICE_URL/transcribe" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"gcs_uri": "gs://my-bucket/meetings/2026-08-05.m4a", "response_format": "srt"}'
```

The runtime service account is granted `roles/storage.objectViewer` by the deploy script. Tighten it to a single bucket if you prefer:

```bash
gcloud projects remove-iam-policy-binding YOUR_PROJECT \
  --member=serviceAccount:qwen3-asr-sa@YOUR_PROJECT.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer

gcloud storage buckets add-iam-policy-binding gs://my-bucket \
  --member=serviceAccount:qwen3-asr-sa@YOUR_PROJECT.iam.gserviceaccount.com \
  --role=roles/storage.objectViewer
```

### File upload (OpenAI-compatible)

```bash
curl -X POST "$SERVICE_URL/v1/audio/transcriptions" \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@meeting.m4a \
  -F response_format=srt
```

Works with the OpenAI SDKs — point `base_url` at `$SERVICE_URL/v1`:

```python
from openai import OpenAI

client = OpenAI(base_url=f"{SERVICE_URL}/v1", api_key=TOKEN)
with open("meeting.m4a", "rb") as f:
    print(client.audio.transcriptions.create(model="qwen3-asr", file=f).text)
```

### Migrating an existing RunPod client

`POST /runsync` returns the RunPod envelope, so a client only changes its base URL and auth header:

```json
{ "id": "...", "status": "COMPLETED", "output": { "text": "...", "language": "English" } }
```

### Endpoint reference

| Endpoint | Purpose |
| --- | --- |
| `POST /transcribe` | Native. Body is the `input` schema, bare or wrapped. |
| `POST /v1/audio/transcriptions` | OpenAI-compatible multipart upload. |
| `POST /runsync`, `POST /run` | RunPod-shaped response envelope. |
| `GET /health` | Liveness. Never touches the model. |
| `GET /ready` | Readiness. 503 until weights are on the GPU. |
| `POST /warmup` | Blocks until loaded. For post-deploy or keep-warm pings. |
| `GET /docs` | Generated OpenAPI docs. |

---

## Authentication

**IAM (default, recommended).** The service is deployed `--no-allow-unauthenticated`. Grant callers explicitly:

```bash
# A human
gcloud run services add-iam-policy-binding qwen3-asr --region us-central1 \
  --member=user:you@example.com --role=roles/run.invoker

# Another service
gcloud run services add-iam-policy-binding qwen3-asr --region us-central1 \
  --member=serviceAccount:caller@YOUR_PROJECT.iam.gserviceaccount.com \
  --role=roles/run.invoker
```

**Shared secret.** For a public service, or a client that can't mint Google identity tokens, set `QWEN_AUTH_TOKEN` and send it as `Authorization: Bearer <token>` or `X-API-Key`. It's compared in constant time.

```bash
echo -n "$(openssl rand -hex 24)" | gcloud secrets create qwen3-asr-token --data-file=-

gcloud run services update qwen3-asr --region us-central1 \
  --update-secrets=QWEN_AUTH_TOKEN=qwen3-asr-token:latest
```

Leave it unset to disable that check and rely on IAM alone. Don't deploy `--allow-unauthenticated` *without* a token — that's a public GPU anyone can bill you for.

---

## Cold starts and cost

Cloud Run bills per second for as long as an instance exists — through the request, plus a short idle window before it scales down.

A cold start is roughly: **image pull (streamed, ~10–30 s) + model load onto the GPU (~30–60 s)**. The startup probe on `/ready` holds traffic back until the weights are resident, so a caller sees latency, not a failure.

Levers, roughly in order of impact:

| Lever | Effect |
| --- | --- |
| `MIN_INSTANCES=0` (default) | Cheapest. Pay only for real work. Every idle gap costs a cold start. |
| `MIN_INSTANCES=1` | No cold starts. You pay for an idle L4 around the clock — only worth it under steady traffic. |
| `QWEN_ENABLE_ALIGNER=false` | Skips the 1.8 GB aligner: smaller image, faster load, ~2 GB less VRAM. Disables timestamps, SRT, and VTT. |
| `--build-arg MODEL_ID=Qwen/Qwen3-ASR-0.6B-hf` | ~⅓ the weights, faster load, lower accuracy. |
| `gs://` or URL input over base64 | Base64 inflates the payload ~33% and the whole thing must arrive before work starts. |
| Batch requests | Amortizes one cold start across many clips. |

Check current rates on the [Cloud Run pricing page](https://cloud.google.com/run/pricing) — GPU, CPU, and memory are billed separately, and rates change.

> **Scheduled warm-up.** If your traffic is predictable (say, business hours), Cloud Scheduler hitting `POST /warmup` is much cheaper than `MIN_INSTANCES=1`:
>
> ```bash
> gcloud scheduler jobs create http qwen3-asr-warmup \
>   --location us-central1 --schedule "*/10 8-18 * * 1-5" \
>   --uri "$SERVICE_URL/warmup" --http-method POST \
>   --oidc-service-account-email qwen3-asr-sa@YOUR_PROJECT.iam.gserviceaccount.com
> ```

---

## The 60-minute ceiling

**Cloud Run's maximum request timeout is 3600 seconds, and it cannot be raised.**

Qwen3-ASR is autoregressive, so realtime factors land in the single digits to low tens. In practice a synchronous request comfortably handles audio up to roughly **45 minutes**; beyond that you're gambling against the timeout, and a timeout loses all the work.

`QWEN_MAX_AUDIO_DURATION_S` still defaults to 4 hours. Lower it to fail fast rather than time out:

```bash
gcloud run services update qwen3-asr --region us-central1 \
  --set-env-vars QWEN_MAX_AUDIO_DURATION_S=2700
```

For genuinely long audio, either:

- **Split client-side** into <30-minute pieces, transcribe in parallel (raise `MAX_INSTANCES`), and stitch. Parallelism also makes it *faster* than one long request.
- **Go asynchronous** — have the caller enqueue to Cloud Tasks or Pub/Sub, and run the same image as a **Cloud Run job** (jobs allow up to 24 hours) writing results to GCS. The container needs no changes; only the trigger differs.

---

## Configuration

Every `QWEN_*` variable from the [README](README.md#environment-variables) works identically here. Cloud Run adds:

| Variable | Default | Purpose |
| --- | --- | --- |
| `QWEN_PRELOAD_MODEL` | `true` | Load weights during startup instead of on the first request. |
| `QWEN_MAX_CONCURRENCY` | `1` | In-flight inferences per instance. Keep at 1 unless you know why not. |
| `QWEN_QUEUE_TIMEOUT_S` | `900` | How long a request waits for a free slot before 503. |
| `QWEN_AUTH_TOKEN` | *(unset)* | Shared-secret bearer token. Unset = IAM only. |
| `QWEN_WORK_DIR` | *(unset)* | Scratch dir. Defaults to `/tmp`, which is **memory-backed** on Cloud Run. |
| `PORT` | `8080` | Set by Cloud Run. |

Update without rebuilding:

```bash
gcloud run services update qwen3-asr --region us-central1 \
  --set-env-vars QWEN_ENABLE_ALIGNER=false,QWEN_MAX_NEW_TOKENS=1024
```

### Sizing memory

`/tmp` is a tmpfs on Cloud Run, so **scratch files count against `--memory`.** A 4-hour transcode is a ~460 MB WAV, and chunking writes overlapping windows alongside it — peak scratch usage is roughly 2.5× the WAV. The default 32Gi absorbs this easily; if you trim memory, trim `QWEN_MAX_AUDIO_DURATION_S` with it.

### Declarative deploys

`gcloud run deploy` flags can't express startup or liveness probes. For those, use the service spec:

```bash
sed "s/PROJECT_ID/$(gcloud config get-value project)/g" \
  deploy/gcp/service.yaml > /tmp/service.yaml
gcloud run services replace /tmp/service.yaml --region us-central1
```

---

## Operations

```bash
# Live logs
gcloud beta run services logs tail qwen3-asr --region us-central1

# Errors only — the app emits structured JSON, so severity filtering works
gcloud logging read \
  'resource.type=cloud_run_revision AND resource.labels.service_name=qwen3-asr AND severity>=ERROR' \
  --limit 50 --format json

# Roll back
gcloud run services update-traffic qwen3-asr --region us-central1 --to-revisions PREVIOUS=100
```

Every transcription logs one structured line with `request_id`, `seconds`, and `audio_seconds`, so you can chart realtime factor straight from Cloud Logging.

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Quota 'TotalNvidiaL4GpuAllocation...' exceeded` | GPU quota is 0. Request it (see Prerequisites) — this is the most common blocker. |
| Deploy: `GPU is not supported in region X` | Pick a GPU region: `us-central1`, `us-east4`, `europe-west1`, `europe-west4`, `asia-southeast1`, `asia-south1`. |
| Build times out | Raise `timeout` in `deploy/gcp/cloudbuild.yaml`. Weight download dominates. |
| Build: `no space left on device` | Raise `options.diskSizeGb` in `cloudbuild.yaml`. |
| Container fails to start | Almost always the startup probe timing out during model load. Check logs for `model load failed`. |
| First request takes ~90 s | Expected cold start. Use `POST /warmup`, or `MIN_INSTANCES=1`. |
| `503 Server busy` | All inference slots in use. Raise `MAX_INSTANCES` — not `QWEN_MAX_CONCURRENCY`. |
| `504` after exactly 3600 s | Cloud Run's hard timeout. See [the 60-minute ceiling](#the-60-minute-ceiling). |
| `403 Forbidden` | Caller lacks `roles/run.invoker`, or the identity token expired (they last 1 hour). |
| `Could not read gs://...` | Runtime service account lacks `roles/storage.objectViewer` on that bucket. |
| Memory exceeded on long audio | `/tmp` is RAM. Raise `--memory` or lower `QWEN_MAX_AUDIO_DURATION_S`. |
| `CUDA out of memory` | Shouldn't happen on a 24 GB L4. If it does, set `QWEN_ENABLE_ALIGNER=false`. |

---

## Teardown

```bash
gcloud run services delete qwen3-asr --region us-central1
gcloud artifacts repositories delete qwen3-asr --location us-central1
gcloud iam service-accounts delete qwen3-asr-sa@YOUR_PROJECT.iam.gserviceaccount.com
```

Deleting the service stops all charges — there is nothing left running at zero traffic. The Artifact Registry image keeps costing storage until deleted.
