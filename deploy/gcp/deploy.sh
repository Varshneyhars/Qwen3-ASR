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
# Projects created from ~2024 on no longer get the legacy
# PROJECT_NUMBER@cloudbuild.gserviceaccount.com identity, and `gcloud builds
# submit` then fails with PERMISSION_DENIED even for a project owner. Running
# builds as an explicit service account works on old and new projects alike.
BUILD_SA_NAME="${BUILD_SA_NAME:-qwen3-asr-build}"

# L4 is the only GPU Cloud Run offers. It requires >=4 CPU and >=16Gi; 8/32
# gives ffmpeg room to work and headroom for /tmp, which is a memory-backed
# tmpfs on Cloud Run — a 4-hour WAV plus its chunks lives there.
GPU_TYPE="${GPU_TYPE:-nvidia-l4}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-32Gi}"
MAX_INSTANCES="${MAX_INSTANCES:-12}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"    # 0 = scale to zero, at the cost of cold starts

# CONCURRENCY x MAX_INSTANCES is how many requests may be in flight at once, and
# callers size their own parallelism from it: 3 x 12 = 36. GPU_SLOTS is separate
# and smaller on purpose. Cloud Run accepting three requests per instance keeps
# them queued at the instance instead of refused, while GPU_SLOTS=1 keeps one
# forward pass running at a time, which is what a single L4 actually wants.
# Raising GPU_SLOTS without measuring trades throughput for CUDA OOM risk.
CONCURRENCY="${CONCURRENCY:-3}"
GPU_SLOTS="${GPU_SLOTS:-1}"

# Clips per GPU forward pass. The processor pads every item in a pass to the
# longest one, which is why asr_core sorts a batch by duration before grouping.
GPU_BATCH_SIZE="${GPU_BATCH_SIZE:-32}"
# Items accepted in one /batch request. Callers pack to this, so it is also the
# number the ingest's own batch size is pinned to.
MAX_BATCH_ITEMS="${MAX_BATCH_ITEMS:-32}"
# Downloads, transcodes and duration probes running at once inside one request.
# Matches CPU below; this prefix used to run serially and dominated a big batch.
FETCH_WORKERS="${FETCH_WORKERS:-8}"
TIMEOUT="${TIMEOUT:-3600}"             # Cloud Run's ceiling is 60 minutes
ALLOW_UNAUTH="${ALLOW_UNAUTH:-false}"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"
# A full address in SERVICE_ACCOUNT wins over SERVICE_ACCOUNT_NAME, because the
# identity a service already runs as is not always one this script could name: a
# service deployed by hand often runs as the default
# PROJECT_NUMBER-compute@developer.gserviceaccount.com, which is a different
# domain entirely. Redeploying such a service must not quietly move it onto a
# new identity with different permissions, so pass the one it has.
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"
OWN_SERVICE_ACCOUNT="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUILD_SA="${BUILD_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "ERROR: no project set. Run 'gcloud config set project YOUR_PROJECT' or pass PROJECT_ID=..." >&2
  exit 1
fi

echo "==> Project ${PROJECT_ID} / region ${REGION} / service ${SERVICE}"
echo "==> Image ${IMAGE}"
echo "==> ${CONCURRENCY} req/instance x ${MAX_INSTANCES} instances = $((CONCURRENCY * MAX_INSTANCES)) slots"
echo "==> ${GPU_SLOTS} GPU slot(s)/instance, ${GPU_BATCH_SIZE} clips/pass, ${MAX_BATCH_ITEMS} items/request, ${FETCH_WORKERS} parallel fetches"

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

if [[ "${SERVICE_ACCOUNT}" != "${OWN_SERVICE_ACCOUNT}" ]]; then
  # Somebody else's identity. Creating it or editing its roles is not this
  # script's business, so it is used exactly as given.
  echo "==> Runtime service account ${SERVICE_ACCOUNT} (supplied, left alone)"
else
  echo "==> Ensuring runtime service account '${SERVICE_ACCOUNT}'"
  gcloud iam service-accounts describe "${SERVICE_ACCOUNT}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --display-name "Qwen3-ASR Cloud Run runtime" \
    --project "${PROJECT_ID}"

  # Needed only for gs:// audio inputs, so a failure here (usually because the
  # deployer isn't a project owner) must not sink an otherwise fine deploy.
  if ! gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
        --member "serviceAccount:${SERVICE_ACCOUNT}" \
        --role roles/storage.objectViewer \
        --condition=None >/dev/null 2>&1; then
    echo "    WARNING: could not grant roles/storage.objectViewer to ${SERVICE_ACCOUNT}."
    echo "             Everything works except gs:// audio inputs. Grant it later with:"
    echo "               gcloud projects add-iam-policy-binding ${PROJECT_ID} \\"
    echo "                 --member=serviceAccount:${SERVICE_ACCOUNT} --role=roles/storage.objectViewer"
  fi
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "==> Ensuring build service account '${BUILD_SA}'"
  gcloud iam service-accounts describe "${BUILD_SA}" --project "${PROJECT_ID}" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "${BUILD_SA_NAME}" \
    --display-name "Qwen3-ASR Cloud Build" \
    --project "${PROJECT_ID}"

  # Least privilege for a build: push the image, write logs, read the uploaded
  # source tarball. Deliberately not roles/editor.
  for role in roles/artifactregistry.writer roles/logging.logWriter roles/storage.objectAdmin; do
    gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
      --member "serviceAccount:${BUILD_SA}" \
      --role "${role}" --condition=None >/dev/null
  done

  echo "==> Building image (this takes 15-30 min; it downloads ~5 GB of weights)"
  # --default-buckets-behavior is required alongside --service-account: the
  # legacy global staging bucket isn't usable with a user-specified identity.
  gcloud builds submit \
    --config deploy/gcp/cloudbuild.yaml \
    --project "${PROJECT_ID}" \
    --service-account "projects/${PROJECT_ID}/serviceAccounts/${BUILD_SA}" \
    --default-buckets-behavior=regional-user-owned-bucket \
    --substitutions "_REGION=${REGION},_REPO=${REPO},_IMAGE=${IMAGE_NAME},_TAG=${TAG}" \
    .
else
  echo "==> SKIP_BUILD=1, using existing ${IMAGE}"
fi

AUTH_FLAG="--no-allow-unauthenticated"
[[ "${ALLOW_UNAUTH}" == "true" ]] && AUTH_FLAG="--allow-unauthenticated"

# The startup probe is not optional. Without it Cloud Run routes traffic the
# moment the port opens — while the model is still loading — and a burst of
# cold requests comes back as 500s and "Rate exceeded" instead of queueing.
# 24 x 10s = 240s, Cloud Run's maximum startup budget.
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
  --set-env-vars "QWEN_PRELOAD_MODEL=true,QWEN_ENABLE_ALIGNER=true,QWEN_PRECISION=bfloat16,QWEN_MAX_CONCURRENCY=${GPU_SLOTS},QWEN_GPU_BATCH_SIZE=${GPU_BATCH_SIZE},QWEN_MAX_BATCH_ITEMS=${MAX_BATCH_ITEMS},QWEN_FETCH_WORKERS=${FETCH_WORKERS}" \
  --startup-probe "httpGet.path=/ready,initialDelaySeconds=5,periodSeconds=10,timeoutSeconds=5,failureThreshold=24" \
  --liveness-probe "httpGet.path=/health,periodSeconds=30,timeoutSeconds=5,failureThreshold=3" \
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
