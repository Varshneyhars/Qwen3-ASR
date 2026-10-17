"""HTTP server for Qwen3-ASR on Cloud Run (GPU).

Cloud Run has no job queue — it routes HTTP requests straight at the container,
so this wraps asr_core.transcribe in the three surfaces callers actually want:

    POST /transcribe                  native schema (same as the RunPod input)
    POST /batch                       many clips per request, results tagged by id
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

# How many requests may be inside asr_core at once. This is NOT the GPU limit.
#
# asr_core owns that (asr_core.GPU_SLOTS) and holds it only around a forward
# pass, so a request waiting for the GPU is still free to download, transcode
# and probe its audio. This number is larger than GPU_SLOTS on purpose, by
# QWEN_PREFETCH_REQUESTS: while one request is on the GPU, the next ones are
# already fetching, and the GPU gets its next batch handed to it ready.
#
# It used to be the GPU limit, wrapped around the whole of asr_core.transcribe.
# That meant the GPU sat idle through the next request's entire prefix: 32
# downloads, transcodes and duration probes, every request, with the device
# doing nothing. Prefetching is the single change that stops that.
PREFETCH_REQUESTS = max(0, int(os.environ.get("QWEN_PREFETCH_REQUESTS", "2")))
ADMIT_CONCURRENCY = asr_core.GPU_SLOTS + PREFETCH_REQUESTS
_admit = threading.BoundedSemaphore(ADMIT_CONCURRENCY)

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
         aligner=asr_core.ENABLE_ALIGNER, gpu_slots=asr_core.GPU_SLOTS,
         admit=ADMIT_CONCURRENCY, fetch_workers=asr_core.FETCH_WORKERS,
         gpu_batch=asr_core.GPU_BATCH_SIZE, max_batch_items=asr_core.MAX_BATCH_ITEMS)
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
        "endpoints": ["/transcribe", "/batch", "/v1/audio/transcriptions",
                      "/runsync", "/health", "/ready", "/warmup", "/docs"],
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
    """Admit a request into asr_core, on a worker thread.

    The GPU is gated inside asr_core, not here, so this only caps how many
    requests may be fetching and transcoding at the same time. 503 rather than a
    long silence once even that is full, because a caller can retry a 503 and
    cannot do anything with a connection that simply never answers.
    """
    acquired = _admit.acquire(timeout=float(os.environ.get("QWEN_QUEUE_TIMEOUT_S", "900")))
    if not acquired:
        raise HTTPException(status_code=503, detail="Server busy; all inference slots are in use.")
    try:
        return asr_core.transcribe(payload)
    finally:
        _admit.release()


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
#                                 Batch route                                  #
# --------------------------------------------------------------------------- #
# The shape the annotation platform's ingest already speaks. It was serving a
# separate 168-line Flask prototype that never lived in this repo; folding the
# route in here is what lets the two converge on one codebase instead of
# drifting further apart.
#
#   request   {"items": [{"id": "...", "gcs_uri": "https://... | gs://..."}],
#              "language": "Italian"}
#   response  {"results": [{"id": ..., "text": ..., "language": ...}],
#              "errors":  [{"id": ..., "error": ...}],
#              "n": int, "elapsed_s": float, "per_item_ms": float}
#
# Results carry the id they were asked about, so nothing is matched by
# position once it leaves here.

# The prototype hardcoded language="English" on every model call and never read
# the caller's. English callers therefore depend on English being the default,
# while every other locale was being transcribed as if it were English. Keeping
# the default means English behaviour does not move; reading the field means
# the other locales finally get the language they asked for.
BATCH_DEFAULT_LANGUAGE = os.environ.get("QWEN_BATCH_DEFAULT_LANGUAGE", "English")

# Where a batch item's audio may be named. "gcs_uri" first because that is what
# the ingest sends, and despite the name it carries a presigned https URL as
# often as a gs:// one.
_ITEM_AUDIO_KEYS = ("gcs_uri", "url", "audio_url", "audio", "audio_base64")


def _batch_item_source(item: Dict[str, Any], where: str) -> str:
    for key in _ITEM_AUDIO_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise HTTPException(
        status_code=400,
        detail=f"{where}: need one of {', '.join(_ITEM_AUDIO_KEYS)}.",
    )


def _batch_prompt(items: list, payload: Dict[str, Any]) -> Optional[str]:
    """One prompt for the whole request.

    asr_core biases a forward pass, not an item, so a batch cannot carry a
    different context per clip. Identical per-item contexts collapse to the one
    they all agree on; differing ones are refused rather than silently applied
    to clips that did not ask for them.
    """
    shared = payload.get("prompt") or payload.get("context") or payload.get("hotwords")
    contexts = {
        c.strip()
        for c in (item.get("context") for item in items)
        if isinstance(c, str) and c.strip()
    }
    if not contexts:
        return shared
    if len(contexts) > 1:
        raise HTTPException(
            status_code=400,
            detail="Per-item 'context' must be identical across a batch; a GPU "
                   "pass is biased as a whole. Send one request-level 'prompt' "
                   "instead, or split the batch.",
        )
    return contexts.pop()


@app.post("/batch")
async def batch_endpoint(request: Request):
    """Transcribe many clips in one request, each result tagged with its id."""
    _check_auth(request)
    payload = await _read_json(request)
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex

    raw = payload.get("items")
    if raw is None:
        raw = payload.get("sources")
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=400, detail="items[] required")

    items: list = []
    for i, entry in enumerate(raw):
        if isinstance(entry, str):
            entry = {"gcs_uri": entry}
        if not isinstance(entry, dict):
            raise HTTPException(status_code=400, detail=f"items[{i}] must be an object or a string.")
        items.append(entry)

    sources = [_batch_item_source(it, f"items[{i}]") for i, it in enumerate(items)]
    # Falling back to the source means a caller that sends no ids still gets
    # results it can match up, rather than an index it has to trust.
    ids = [
        it["id"] if isinstance(it.get("id"), str) and it["id"].strip() else sources[i]
        for i, it in enumerate(items)
    ]

    language = payload.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise HTTPException(status_code=400, detail="'language' must be a non-empty string, or omitted.")

    inner: Dict[str, Any] = {
        "audio": sources,
        "language": (language or BATCH_DEFAULT_LANGUAGE).strip(),
        "timestamps": bool(payload.get("timestamps", False)),
    }
    prompt = _batch_prompt(items, payload)
    if prompt:
        inner["prompt"] = prompt
    if payload.get("max_new_tokens") is not None:
        inner["max_new_tokens"] = payload["max_new_tokens"]

    started = time.perf_counter()
    result = await _transcribe_async(inner, request_id)

    # asr_core flattens a one-item batch into the single-item shape, so put it
    # back into a list before anything reads results[i].
    if "results" in result:
        per_item = result["results"]
    elif "error" in result:
        # Whole-request failure: bad input, or the model never loaded. 400 keeps
        # a caller from retrying a malformed body forever.
        return JSONResponse(status_code=400, content={"error": result["error"]})
    else:
        per_item = [result]

    results, errors = [], []
    for i, item_result in enumerate(per_item):
        item_id = ids[i] if i < len(ids) else None
        if not isinstance(item_result, dict) or "error" in (item_result or {}):
            errors.append({"id": item_id, "error": (item_result or {}).get("error", "no result")})
            continue
        row: Dict[str, Any] = {
            "id": item_id,
            "text": item_result.get("text", ""),
            "language": item_result.get("language"),
        }
        if inner["timestamps"] and item_result.get("timestamps"):
            row["timestamps"] = item_result["timestamps"]
        if item_result.get("duration") is not None:
            row["duration"] = item_result["duration"]
        results.append(row)

    # An item the model skipped entirely would otherwise vanish; the caller must
    # see one entry per clip it asked about.
    for missing in ids[len(per_item):]:
        errors.append({"id": missing, "error": "internal: item produced no result"})

    elapsed = time.perf_counter() - started
    _log("INFO", "batch complete", request_id=request_id, items=len(ids),
         ok=len(results), failed=len(errors), seconds=round(elapsed, 3),
         language=inner["language"])
    return JSONResponse(
        content={
            "results": results,
            "errors": errors,
            "n": len(results),
            "elapsed_s": round(elapsed, 2),
            "per_item_ms": round(elapsed * 1000 / max(len(ids), 1), 1),
        },
        headers={"x-request-id": request_id},
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
