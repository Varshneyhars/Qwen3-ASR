#!/usr/bin/env bash
# Exercise a deployed Cloud Run service end to end.
#
#   ./deploy/gcp/smoke_test.sh                       # resolves the URL via gcloud
#   ./deploy/gcp/smoke_test.sh https://my-svc.run.app
#
# Uses an identity token by default (matching --no-allow-unauthenticated).
# For a public service with a shared secret, pass QWEN_AUTH_TOKEN=... instead.
set -euo pipefail

REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-qwen3-asr}"
URL="${1:-}"

if [[ -z "${URL}" ]]; then
  URL="$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format 'value(status.url)')"
fi
if [[ -z "${URL}" ]]; then
  echo "ERROR: could not resolve the service URL. Pass it as the first argument." >&2
  exit 1
fi

if [[ -n "${QWEN_AUTH_TOKEN:-}" ]]; then
  TOKEN="${QWEN_AUTH_TOKEN}"
else
  TOKEN="$(gcloud auth print-identity-token)"
fi
AUTH=(-H "Authorization: Bearer ${TOKEN}")

SAMPLE="${SAMPLE_URL:-https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav}"
fail=0

check() {
  local name="$1" expected="$2" got="$3"
  if [[ "${got}" == "${expected}" ]]; then
    echo "  PASS  ${name} (${got})"
  else
    echo "  FAIL  ${name} (expected ${expected}, got ${got})"
    fail=1
  fi
}

echo "==> ${URL}"

echo "[1/7] liveness"
code="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" "${URL}/health")"
check "GET /health" 200 "${code}"

echo "[2/7] readiness (waits up to 5 min for a cold start)"
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" "${URL}/ready")"
  [[ "${code}" == "200" ]] && break
  sleep 5
done
check "GET /ready" 200 "${code}"

echo "[3/7] transcription"
body="$(curl -s "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"audio_url\":\"${SAMPLE}\",\"response_format\":\"text\"}" \
  "${URL}/transcribe")"
echo "      ${body}" | head -c 400; echo
if grep -qi 'quilter' <<<"${body}"; then
  echo "  PASS  POST /transcribe returned the expected transcript"
else
  echo "  FAIL  POST /transcribe did not contain the expected text"
  fail=1
fi

echo "[4/7] subtitles + timestamps"
body="$(curl -s "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"audio_url\":\"${SAMPLE}\",\"response_format\":\"srt\"}" \
  "${URL}/transcribe")"
if grep -q -- '-->' <<<"${body}"; then
  echo "  PASS  SRT contains cue timings"
else
  echo "  FAIL  SRT had no cues (is the aligner enabled?)"
  fail=1
fi

echo "[5/7] input validation rejects a bad request"
code="$(curl -s -o /dev/null -w '%{http_code}' "${AUTH[@]}" \
  -H 'Content-Type: application/json' -d '{"response_format":"text"}' \
  "${URL}/transcribe")"
check "POST /transcribe (no audio)" 400 "${code}"

echo "[6/7] POST /batch returns a result per id"
body="$(curl -s "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"items\":[{\"id\":\"smoke-a\",\"gcs_uri\":\"${SAMPLE}\"},{\"id\":\"smoke-b\",\"gcs_uri\":\"${SAMPLE}\"}]}" \
  "${URL}/batch")"
# Both ids must come back, because a caller matches on id and never on position.
if grep -q '"id": *"smoke-a"' <<<"${body}" && grep -q '"id": *"smoke-b"' <<<"${body}"; then
  echo "  PASS  both ids echoed on their own result"
else
  echo "  FAIL  /batch did not answer for both ids: ${body:0:300}"
  fail=1
fi

echo "[7/7] POST /batch isolates one unreachable item"
body="$(curl -s "${AUTH[@]}" -H 'Content-Type: application/json' \
  -d "{\"items\":[{\"id\":\"good\",\"gcs_uri\":\"${SAMPLE}\"},{\"id\":\"bad\",\"gcs_uri\":\"https://example.invalid/nope.wav\"}]}" \
  "${URL}/batch")"
# One bad URL must cost one clip, not the request. The prototype this replaced
# let the exception escape and returned 500 for every clip in the batch.
if grep -q '"id": *"good"' <<<"${body}" && grep -q '"id": *"bad"' <<<"${body}"; then
  echo "  PASS  the good clip still returned, the bad one came back as an error"
else
  echo "  FAIL  a single bad URL took the batch with it: ${body:0:300}"
  fail=1
fi

echo
if [[ "${fail}" -eq 0 ]]; then
  echo "All checks passed."
else
  echo "Some checks failed." >&2
fi
exit "${fail}"
