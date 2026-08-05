#!/usr/bin/env bash
# One-shot deploy of Qwen3-ASR to Cloud Run with a GPU.
#
#   ./deploy/gcp/deploy.sh                      # build + deploy
#   SKIP_BUILD=1 ./deploy/gcp/deploy.sh         # redeploy an existing image
#
# Override anything below with an env var, e.g.
#   PROJECT_ID=my-proj REGION=europe-west4 ./deploy/gcp/deploy.sh
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-qwen3-asr}"
REPO="${REPO:-qwen3-asr}"
IMAGE_NAME="${IMAGE_NAME:-qwen3-asr}"
TAG="${TAG:-v1}"
SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-qwen3-asr-sa}"

# L4 is the only GPU Cloud Run offers. It requires >=4 CPU and >=16Gi; 8/32
# gives ffmpeg room to work and headroom for /tmp, which is a memory-backed
# tmpfs on Cloud Run — a 4-hour WAV plus its chunks lives there.
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-32Gi}"
MAX_INSTANCES="${MAX_INSTANCES:-3}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"    # 0 = scale to zero, at the cost of cold starts
CONCURRENCY="${CONCURRENCY:-1}"        # one GPU, one in-flight request
TIMEOUT="${TIMEOUT:-3600}"             # Cloud Run's ceiling is 60 minutes
ALLOW_UNAUTH="${ALLOW_UNAUTH:-false}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: no project set. Run 'gcloud config set project YOUR_PROJECT' or pass PROJECT_ID=..." >&2
  exit 1
fi

echo "==> Project ${PROJECT_ID} / region ${REGION} / service ${SERVICE}"
echo "==> Image ${IMAGE}"

echo "==> Enabling APIs"
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  --project "${PROJECT_ID}"

echo "==> Ensuring Artifact Registry repo '${REPO}'"
gcloud artifacts repositories describe "${REPO}" \
  --location "${REGION}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location "${REGION}" \
  --description "Qwen3-ASR Cloud Run images" \
  --project "${PROJECT_ID}"

echo "==> Ensuring runtime service account '${SERVICE_ACCOUNT}'"
gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
  --display-name "Qwen3-ASR Cloud Run runtime" \
  --project "${PROJECT_ID}"

# Needed only for gs:// audio inputs. Harmless otherwise, and scoped to read.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SERVICE_ACCOUNT}" \
  --role roles/storage.objectViewer \
  --condition=None >/dev/null

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "==> Building image (this takes 15-30 min; it downloads ~5 GB of weights)"
  gcloud builds submit \
    --config deploy/gcp/cloudbuild.yaml \
    --project "${PROJECT_ID}" \
    --substitutions "_REGION=${REGION},_REPO=${REPO},_IMAGE=${IMAGE_NAME},_TAG=${TAG}" \
    .
else
  echo "==> SKIP_BUILD=1, using existing ${IMAGE}"
fi

AUTH_FLAG="--no-allow-unauthenticated"
[[ "${ALLOW_UNAUTH}" == "true" ]] && AUTH_FLAG="--allow-unauthenticated"

echo "==> Deploying to Cloud Run"
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --platform managed \
  --execution-environment gen2 \
  --gpu 1 \
  --gpu-type "${GPU_TYPE}" \
  --no-gpu-zonal-redundancy \
  --cpu "${CPU}" \
  --memory "${MEMORY}" \
  --min-instances "${MIN_INSTANCES}" \
  --max-instances "${MAX_INSTANCES}" \
  --concurrency "${CONCURRENCY}" \
  --timeout "${TIMEOUT}" \
  --port 8080 \
  --service-account "${SERVICE_ACCOUNT}" \
  --no-cpu-throttling \
  --set-env-vars "QWEN_PRELOAD_MODEL=true,QWEN_ENABLE_ALIGNER=true,QWEN_PRECISION=bfloat16,QWEN_MAX_CONCURRENCY=${CONCURRENCY}" \
  ${AUTH_FLAG}

URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" \
        --project "${PROJECT_ID}" --format 'value(status.url)')"

echo
echo "==> Deployed: ${URL}"
if [[ "${ALLOW_UNAUTH}" == "true" ]]; then
  echo "    Service is PUBLIC. Set QWEN_AUTH_TOKEN to require a bearer token:"
  echo "      gcloud run services update ${SERVICE} --region ${REGION} --set-env-vars QWEN_AUTH_TOKEN=\$(openssl rand -hex 24)"
else
  echo "    Service requires IAM auth. Grant a caller:"
  echo "      gcloud run services add-iam-policy-binding ${SERVICE} --region ${REGION} \\"
  echo "        --member=user:you@example.com --role=roles/run.invoker"
fi
echo
echo "    Smoke test:"
echo "      ./deploy/gcp/smoke_test.sh ${URL}"
