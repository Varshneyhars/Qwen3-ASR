"""HTTP server for Qwen3-ASR on Cloud Run (GPU).

Cloud Run has no job queue — it routes HTTP requests straight at the container,
so this wraps asr_core.transcribe in the three surfaces callers actually want:

    POST /transcribe                  native schema (same as the RunPod input)
    POST /v1/audio/transcriptions     OpenAI-compatible multipart upload
    POST /runsync                     RunPod-shaped envelope, for easy migration

Plus GET /health (liveness, never blocks) and GET /ready (readiness, reports
whether the weights are resident).
"""
import asyncio
import base64
import json
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

import asr_core

# One GPU, one model instance. Concurrent generate() calls on the same module
# would race on CUDA memory and give no throughput win, so inference is
# serialized here and Cloud Run is deployed with --concurrency 1 to match.
MAX_CONCURRENCY = max(1, int(os.environ.get("QWEN_MAX_CONCURRENCY", "1")))
_gpu_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)

# Load the weights during startup rather than inside the first user request.
# Cloud Run grants a CPU boost while an instance is starting, so this is both
# faster and not billed against someone's request latency.
PRELOAD = os.environ.get("QWEN_PRELOAD_MODEL", "true").lower() == "true"

# Optional shared secret, for when the service is deployed with
# --allow-unauthenticated and IAM isn't doing the gatekeeping.
AUTH_TOKEN = os.environ.get("QWEN_AUTH_TOKEN", "").strip()

REVISION = os.environ.get("K_REVISION", "local")
SERVICE = os.environ.get("K_SERVICE", "qwen3-asr")

_state: Dict[str, Any] = {"ready": False, "loading": False, "load_error": None}


def _log(severity: str, message: str, **fields: Any) -> None:
    """Structured log line. Cloud Logging parses JSON on stdout and picks up
    `severity` and `message`, so these show up correctly leveled and filterable
    instead of as an undifferentiated INFO blob."""
    payload = {"severity": severity, "message": message,
               "service": SERVICE, "revision": REVISION, **fields}
    print(json.dumps(payload, default=str), flush=True)


def _load_models() -> None:
    _state["loading"] = True
    try:
        info = asr_core.warmup()
        _state["ready"] = True
        _state["load_error"] = None
        _log("INFO", "models loaded", **info)
    except Exception as e:
        _state["load_error"] = f"{type(e).__name__}: {e}"
        _log("ERROR", "model load failed", error=_state["load_error"])
    finally:
        _state["loading"] = False


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _log("INFO", "starting", model=asr_core.MODEL_ID, preload=PRELOAD,
         aligner=asr_core.ENABLE_ALIGNER, concurrency=MAX_CONCURRENCY)
    if PRELOAD:
        # In a background thread so uvicorn binds $PORT immediately. Cloud Run
        # kills an instance that hasn't opened its port within the startup
        # timeout, and loading ~5 GB of weights would blow straight through it.
        threading.Thread(target=_load_models, name="model-preload", daemon=True).start()
    yield
    _log("INFO", "shutting down")


app = FastAPI(
    title="Qwen3-ASR",
    description="Multilingual speech recognition with Qwen3-ASR on Cloud Run GPU.",
    version="1.0.0",
    docs_url="/docs",
    lifespan=_lifespan,
)


# --------------------------------------------------------------------------- #
#                                    Auth                                      #
# --------------------------------------------------------------------------- #
def _check_auth(request: Request) -> None:
    """Verify the shared secret, when one is configured.

    Cloud Run IAM (--no-allow-unauthenticated) is the primary control; this is
    the fallback for public services. Skipped entirely when unset.
    """
    if not AUTH_TOKEN:
        return
    header = request.headers.get("authorization", "")
    presented = header[7:].strip() if header.lower().startswith("bearer ") else \
        request.headers.get("x-api-key", "").strip()
    # Constant-time compare so a wrong token can't be recovered byte by byte.
    if not presented or not secrets.compare_digest(presented, AUTH_TOKEN):
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


# --------------------------------------------------------------------------- #
#                                   Health                                     #
# --------------------------------------------------------------------------- #
@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": SERVICE,
        "revision": REVISION,
        "model": asr_core.MODEL_ID,
        "aligner": asr_core.ALIGNER_MODEL_ID if asr_core.ENABLE_ALIGNER else None,
        "ready": _state["ready"],
        "endpoints": ["/transcribe", "/v1/audio/transcriptions", "/runsync",
                      "/health", "/ready", "/warmup", "/docs"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness. Deliberately never touches the model — a slow first load is
    not a reason for Cloud Run to recycle the instance."""
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response) -> Dict[str, Any]:
    """Readiness. 503 until the weights are resident, so Cloud Run's startup
    probe holds traffic back instead of racing the load."""
    if _state["load_error"] and not _state["ready"]:
        response.status_code = 503
        return {"status": "error", "error": _state["load_error"]}
    if not (_state["ready"] or asr_core.is_loaded()):
        response.status_code = 503
        return {"status": "loading" if _state["loading"] else "cold"}
    return {"status": "ready", "model": asr_core.model_info()}


@app.post("/warmup")
def warmup_endpoint() -> Dict[str, Any]:
    """Block until the model is loaded. Useful right after a deploy, or from a
    min-instances=0 keep-warm ping."""
    if not asr_core.is_loaded():
        _load_models()
    if _state["load_error"] and not asr_core.is_loaded():
        raise HTTPException(status_code=503, detail=_state["load_error"])
    return {"status": "ready", "model": asr_core.model_info()}


# --------------------------------------------------------------------------- #
#                                 Transcription                                #
# --------------------------------------------------------------------------- #
def _run_blocking(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize GPU work behind the semaphore, on a worker thread."""
    acquired = _gpu_slots.acquire(timeout=float(os.environ.get("QWEN_QUEUE_TIMEOUT_S", "900")))
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy; all inference slots are in use.")
    try:
        return asr_core.transcribe(payload)
    finally:
        _gpu_slots.release()


async def _transcribe_async(payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    started = time.perf_counter()
    # asr_core is synchronous and CPU/GPU bound; off the event loop it goes, so
    # /health stays responsive while a 40-minute file is being transcribed.
    result = await asyncio.to_thread(_run_blocking, payload)
    _log(
        "ERROR" if "error" in result else "INFO",
        "transcribe failed" if "error" in result else "transcribe complete",
        request_id=request_id,
        seconds=round(time.perf_counter() - started, 3),
        audio_seconds=(result.get("metrics") or {}).get("audio_seconds"),
        error=result.get("error"),
    )
    return result


def _unwrap(body: Any) -> Dict[str, Any]:
    """Accept both {"input": {...}} (RunPod shape) and a flat {...} body."""
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    inner = body.get("input")
    return inner if isinstance(inner, dict) else body


async def _read_json(request: Request) -> Dict[str, Any]:
    try:
        return _unwrap(await request.json())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}") from e


@app.post("/transcribe")
async def transcribe_endpoint(request: Request):
    """Native endpoint. Body is the asr_core.transcribe input schema, either
    bare or wrapped in {"input": ...}."""
    _check_auth(request)
    payload = await _read_json(request)
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex

    result = await _transcribe_async(payload, request_id)
    if "error" in result:
        # Bad input, not a server fault — 400 so callers don't retry forever.
        return JSONResponse(status_code=400, content=result)
    return JSONResponse(content=result, headers={"x-request-id": request_id})


@app.post("/runsync")
@app.post("/run")
async def runsync_endpoint(request: Request):
    """RunPod-shaped response envelope.

    Lets a client already pointed at a RunPod endpoint switch to Cloud Run by
    changing only the base URL and the auth header.
    """
    _check_auth(request)
    payload = await _read_json(request)
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex

    result = await _transcribe_async(payload, request_id)
    failed = "error" in result
    return JSONResponse(
        status_code=400 if failed else 200,
        content={
            "id": request_id,
            "status": "FAILED" if failed else "COMPLETED",
            "output": result,
        },
    )


# --------------------------------------------------------------------------- #
#                            OpenAI-compatible route                           #
# --------------------------------------------------------------------------- #
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: Optional[str] = Form(default=None),
):
    """Drop-in replacement for OpenAI's transcription endpoint.

    Takes a multipart upload, so browsers and existing OpenAI SDK clients can
    post a file directly instead of base64-encoding it into JSON.
    """
    _check_auth(request)
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > asr_core.MAX_DOWNLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Upload is {len(raw)} bytes, over the {asr_core.MAX_DOWNLOAD_BYTES} limit.",
        )

    payload: Dict[str, Any] = {
        "audio_base64": base64.b64encode(raw).decode("ascii"),
        "response_format": response_format,
    }
    if language:
        payload["language"] = language
    if prompt:
        payload["prompt"] = prompt
    if timestamp_granularities:
        payload["timestamps"] = True

    result = await _transcribe_async(payload, request_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # OpenAI returns these three as a bare body, not JSON.
    if response_format == "text":
        return PlainTextResponse(result.get("text", ""))
    if response_format == "srt":
        return PlainTextResponse(result.get("srt", ""), media_type="application/x-subrip")
    if response_format == "vtt":
        return PlainTextResponse(result.get("vtt", ""), media_type="text/vtt")
    return JSONResponse(content=result, headers={"x-request-id": request_id})


@app.exception_handler(asr_core.InputError)
async def _input_error_handler(request: Request, exc: asr_core.InputError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        # Cloud Run's front end holds connections open; a short keep-alive makes
        # it reuse a socket the server has already closed, surfacing as random
        # 503s to the caller.
        timeout_keep_alive=int(os.environ.get("KEEP_ALIVE_S", "620")),
        access_log=False,  # _log() already emits a structured line per request
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )
