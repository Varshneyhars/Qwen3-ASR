"""Qwen3-ASR transcription core.

Platform-neutral: no RunPod, no HTTP framework, no GCP SDK at import time.
Both the RunPod worker (handler.py) and the Cloud Run server (server.py) call
transcribe() with the same input dict and get the same output dict back.
"""
import base64
import binascii
import contextlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-ASR-1.7B-hf")
ALIGNER_MODEL_ID = os.environ.get("QWEN_ALIGNER_MODEL_ID", "Qwen/Qwen3-ForcedAligner-0.6B-hf")
# Word timestamps need a second model (~1.8 GB). Disable to shrink the image
# and skip its load if you only ever need plain text.
ENABLE_ALIGNER = os.environ.get("QWEN_ENABLE_ALIGNER", "true").lower() == "true"
PRECISION = os.environ.get("QWEN_PRECISION", "bfloat16").lower()
SAMPLE_RATE = 16000  # Qwen3-ASR expects 16 kHz mono.

MAX_NEW_TOKENS = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "448"))
BATCH_SIZE = int(os.environ.get("QWEN_BATCH_SIZE", "8"))

# Qwen3-ASR processes up to ~5 minutes of audio per forward pass, so anything
# longer must be windowed. This is a hard model constraint, not a tuning knob.
CHUNK_THRESHOLD_S = float(os.environ.get("QWEN_CHUNK_THRESHOLD_S", "280"))
CHUNK_LENGTH_S = float(os.environ.get("QWEN_CHUNK_LENGTH_S", "240"))
CHUNK_OVERLAP_S = float(os.environ.get("QWEN_CHUNK_OVERLAP_S", "5"))

MAX_AUDIO_DURATION_S = float(os.environ.get("QWEN_MAX_AUDIO_DURATION_S", "14400"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("QWEN_MAX_DOWNLOAD_BYTES", str(512 * 1024 * 1024)))
DOWNLOAD_TIMEOUT_S = float(os.environ.get("QWEN_DOWNLOAD_TIMEOUT_S", "120"))
DOWNLOAD_TRIES = max(1, int(os.environ.get("QWEN_DOWNLOAD_TRIES", "3")))

# How many batch items are fetched and transcoded at once. Every item needs a
# download, an ffmpeg transcode and an ffprobe before the GPU sees anything, and
# doing those one after another leaves the GPU idle for the whole prefix: at 32
# items that was most of the request. The work is I/O and subprocesses, so
# threads parallelise it properly despite the GIL. Matches the 8 vCPU an L4
# instance is deployed with; set to 1 to go back to serial.
FETCH_WORKERS = max(1, int(os.environ.get("QWEN_FETCH_WORKERS", "8")))

# GPU slots. Only the forward passes are gated by this, never the fetch, and
# that distinction is the whole point: a request that is waiting its turn on the
# GPU must still be free to download and transcode its audio. Gate the fetch too
# (as this did when the semaphore lived in the HTTP layer) and the GPU goes idle
# for the length of the next request's entire prefix, every single request.
GPU_SLOTS = max(1, int(os.environ.get("QWEN_MAX_CONCURRENCY", "1")))
GPU_QUEUE_TIMEOUT_S = float(os.environ.get("QWEN_QUEUE_TIMEOUT_S", "900"))
_gpu_gate = threading.BoundedSemaphore(GPU_SLOTS)

_fetch_pool: Optional["ThreadPoolExecutor"] = None
_fetch_pool_lock = threading.Lock()
MAX_BATCH_ITEMS = int(os.environ.get("QWEN_MAX_BATCH_ITEMS", "32"))

# How many clips share a single GPU forward pass. Running one clip at a time
# leaves a 1.7B model badly under-occupying an L4 — the GPU spends most of its
# time waiting on memory rather than computing. Feeding several at once is the
# single biggest throughput lever for short audio. Set to 1 to disable.
GPU_BATCH_SIZE = int(os.environ.get("QWEN_GPU_BATCH_SIZE", "8"))
BLOCK_PRIVATE_URLS = os.environ.get("QWEN_BLOCK_PRIVATE_URLS", "true").lower() == "true"

# Scratch space for downloads and transcoded WAVs. On Cloud Run /tmp is a
# tmpfs backed by the instance's memory, so this counts against --memory.
WORK_ROOT = os.environ.get("QWEN_WORK_DIR") or None

# Run a throwaway inference during warmup to pay CUDA's kernel-compilation cost
# up front. Adds ~20s to startup and saves the same on the first real request.
WARMUP_INFERENCE = os.environ.get("QWEN_WARMUP_INFERENCE", "true").lower() == "true"

# Segment grouping for subtitles, derived from word timings.
SEGMENT_MAX_GAP_S = float(os.environ.get("QWEN_SEGMENT_MAX_GAP_S", "0.6"))
SEGMENT_MAX_CHARS = int(os.environ.get("QWEN_SEGMENT_MAX_CHARS", "84"))
SEGMENT_MAX_DURATION_S = float(os.environ.get("QWEN_SEGMENT_MAX_DURATION_S", "6.0"))

VALID_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}
AUDIO_KEYS = ("audio", "audio_url", "audio_base64", "url", "gcs_uri")
_SENTENCE_END = tuple(".?!。？！…")
# A trailing period doesn't end a sentence after these, so they must not break
# a subtitle cue ("Mr. Quilter" should stay one cue, not two).
_ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "jr.", "sr.", "vs.", "etc.",
    "e.g.", "i.e.", "no.", "fig.", "inc.", "ltd.", "co.", "approx.", "dept.",
}
# Below this, a punctuation break is treated as an abbreviation or false
# positive rather than a real sentence end.
SEGMENT_MIN_WORDS = int(os.environ.get("QWEN_SEGMENT_MIN_WORDS", "3"))

# Lazy init: load on first request, not at import. Lets the process reach its
# serving loop instantly so it passes readiness, even when the first model load
# takes 30-90s on a cold disk.
_MODEL = None
_PROCESSOR = None
_ALIGNER = None
_ALIGNER_PROCESSOR = None
_GCS_CLIENT = None
_MODEL_INFO: Dict[str, Any] = {}

# Serializes model loading. Two threads importing transformers at once — a
# background preloader and an early request, say — can catch its lazy module
# system mid-initialization and fail with a spurious
# "cannot import name 'AutoProcessor' from 'transformers'".
# Reentrant because warm-up loads the model and then runs inference through it.
_LOAD_LOCK = threading.RLock()


class InputError(ValueError):
    """Bad job input — caller's fault, not retryable."""


# --------------------------------------------------------------------------- #
#                                   Model                                      #
# --------------------------------------------------------------------------- #
def _dtype():
    import torch

    return {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }.get(PRECISION, torch.bfloat16)


def _get_model():
    """Load the ASR model + processor once per worker process.

    Thread-safe: concurrent callers block until the first finishes rather than
    racing on the transformers import.
    """
    global _MODEL, _PROCESSOR, _MODEL_INFO
    if _MODEL is not None:
        return _MODEL, _PROCESSOR  # fast path: no lock once loaded
    with _LOAD_LOCK:
        if _MODEL is None:
            import torch
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

            started = time.perf_counter()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = _dtype()
            if device == "cpu" and dtype is torch.bfloat16:
                dtype = torch.float32  # bf16 on CPU is unusably slow
            print(f"[lazy-init] Loading {MODEL_ID} (device={device}, dtype={dtype})")

            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = Qwen3ASRForConditionalGeneration.from_pretrained(
                MODEL_ID, dtype=dtype, device_map=device
            )
            model.eval()

            _MODEL_INFO = {
                "model_id": MODEL_ID,
                "device": device,
                "dtype": str(dtype),
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "load_seconds": round(time.perf_counter() - started, 2),
                "aligner_enabled": ENABLE_ALIGNER,
            }
            # Publish only once fully built, so the fast path above can never
            # hand another thread a half-initialized model.
            _PROCESSOR, _MODEL = processor, model
            print(f"[lazy-init] ASR ready in {_MODEL_INFO['load_seconds']}s on {_MODEL_INFO['gpu']}")
    return _MODEL, _PROCESSOR


def _get_aligner():
    """Load the forced aligner. Only touched when timestamps are requested."""
    global _ALIGNER, _ALIGNER_PROCESSOR, _MODEL_INFO
    if not ENABLE_ALIGNER:
        raise InputError(
            "Timestamps require the forced aligner, which is disabled "
            "(QWEN_ENABLE_ALIGNER=false on this endpoint)."
        )
    if _ALIGNER is not None:
        return _ALIGNER, _ALIGNER_PROCESSOR  # fast path: no lock once loaded
    with _LOAD_LOCK:
        if _ALIGNER is None:
            import torch
            from transformers import AutoModelForTokenClassification, AutoProcessor

            started = time.perf_counter()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = _dtype()
            if device == "cpu" and dtype is torch.bfloat16:
                dtype = torch.float32
            print(f"[lazy-init] Loading aligner {ALIGNER_MODEL_ID}")

            processor = AutoProcessor.from_pretrained(ALIGNER_MODEL_ID)
            aligner = AutoModelForTokenClassification.from_pretrained(
                ALIGNER_MODEL_ID, dtype=dtype, device_map=device
            )
            aligner.eval()
            _MODEL_INFO["aligner_load_seconds"] = round(time.perf_counter() - started, 2)
            _ALIGNER_PROCESSOR, _ALIGNER = processor, aligner
            print(f"[lazy-init] Aligner ready in {_MODEL_INFO['aligner_load_seconds']}s")
    return _ALIGNER, _ALIGNER_PROCESSOR


def model_info() -> Dict[str, Any]:
    """Snapshot of what's loaded. Empty dict before the first load."""
    return dict(_MODEL_INFO)


def is_loaded() -> bool:
    return _MODEL is not None


def _warm_kernels() -> None:
    """Run one throwaway inference so CUDA does its kernel autotuning here.

    Loading the weights is not enough. The first generate() on a fresh CUDA
    context pays a large one-off kernel-compilation cost — measured on a Cloud
    Run L4, the first request took 24s for 6s of audio versus ~1.0s once warm.
    Without this, the readiness probe goes green while the service is still
    slow, and the first real caller eats the difference.
    """
    import wave

    started = time.perf_counter()
    tmp = tempfile.mkdtemp(prefix="qwen-warm-", dir=WORK_ROOT)
    try:
        path = os.path.join(tmp, "silence.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * SAMPLE_RATE)  # 1 second of silence

        _transcribe_batch([path], None, None, 8)
        if _ALIGNER is not None:
            # Same story for the aligner's kernels. Aligning silence is
            # meaningless; _align_words swallows the failure either way.
            _align_words(path, "warm up", None)
        print(f"[warmup] kernels warm in {time.perf_counter() - started:.1f}s")
    except Exception as e:
        # A failed warm-up costs latency, not correctness. Never block serving.
        print(f"[warmup] kernel warm-up skipped ({type(e).__name__}: {e})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def warmup(aligner: Optional[bool] = None) -> Dict[str, Any]:
    """Force the model load and CUDA warm-up now instead of on the first request.

    Cloud Run bills for the whole instance lifetime once a GPU is attached, so
    paying the 30-90s load during startup (where CPU boost applies) is strictly
    better than paying it inside a user's request.
    """
    _get_model()
    if ENABLE_ALIGNER if aligner is None else aligner:
        try:
            _get_aligner()
        except Exception as e:  # a broken aligner must not block ASR serving
            print(f"[warmup] aligner load failed ({type(e).__name__}: {e})")
    if WARMUP_INFERENCE:
        _warm_kernels()
    return model_info()


# --------------------------------------------------------------------------- #
#                                   Audio                                      #
# --------------------------------------------------------------------------- #
def _assert_public_host(url: str) -> None:
    """Reject URLs resolving to private/loopback ranges (SSRF guard)."""
    if not BLOCK_PRIVATE_URLS:
        return
    host = urlparse(url).hostname
    if not host:
        raise InputError("URL has no host.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise InputError(f"Could not resolve host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise InputError(
                f"Refusing to fetch {host!r}: resolves to non-public address {ip}. "
                "Set QWEN_BLOCK_PRIVATE_URLS=false if this is intentional."
            )


def _download(url: str, dest: str) -> int:
    """Fetch an http(s) URL to disk, retrying the failures worth retrying.

    A presigned URL hitting a reset connection or a 5xx is a transient fault,
    and without a retry it cost the item its transcript. It used to cost the
    whole request: the prototype this replaced let the exception escape to the
    WSGI layer, so one reset URL returned 500 for every clip in the batch.

    Not retried: the size caps below (the audio will not shrink) and any other
    4xx. An expired or revoked signature returns 403 on every attempt, so
    retrying it only delays the error the caller needs to see.
    """
    _assert_public_host(url)
    last: Optional[Exception] = None
    for attempt in range(DOWNLOAD_TRIES):
        try:
            return _download_once(url, dest)
        except InputError:
            raise
        except (requests.ConnectionError, requests.Timeout,
                requests.exceptions.ChunkedEncodingError) as e:
            last = e
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", 0) or 0
            if status != 429 and not 500 <= status < 600:
                raise
            last = e
        if attempt < DOWNLOAD_TRIES - 1:
            # Short, growing backoff. The caller is holding a GPU slot and the
            # other items in this batch are waiting on the same thread pool, so
            # this is deliberately not a long one.
            wait = 1.5 * (2 ** attempt)
            print(f"[download] attempt {attempt + 1}/{DOWNLOAD_TRIES} failed "
                  f"({type(last).__name__}: {last}); retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Download failed after {DOWNLOAD_TRIES} attempts: "
                       f"{type(last).__name__}: {last}")


def _download_once(url: str, dest: str) -> int:
    written = 0
    with requests.get(
        url, stream=True, timeout=DOWNLOAD_TIMEOUT_S,
        headers={"User-Agent": "qwen3-asr-worker/1.0"},
    ) as resp:
        resp.raise_for_status()
        declared = resp.headers.get("Content-Length")
        if declared and int(declared) > MAX_DOWNLOAD_BYTES:
            raise InputError(f"Audio is {int(declared)} bytes, over the {MAX_DOWNLOAD_BYTES} limit.")
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                written += len(chunk)
                if written > MAX_DOWNLOAD_BYTES:
                    raise InputError(f"Audio exceeded the {MAX_DOWNLOAD_BYTES} byte limit while streaming.")
                f.write(chunk)
    if written == 0:
        raise InputError(f"Downloaded 0 bytes from {url}")
    return written


def _download_gcs(uri: str, dest: str) -> int:
    """Fetch gs://bucket/object using the instance's service account.

    On Cloud Run this needs no credentials in the request — Application Default
    Credentials resolve to the runtime service account, which only needs
    roles/storage.objectViewer on the bucket.
    """
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as e:
        raise InputError(
            "gs:// URIs need the google-cloud-storage package, which is not "
            "installed in this image. Pass an https:// URL (a signed URL works) "
            "or base64 audio instead."
        ) from e

    global _GCS_CLIENT
    path = uri[len("gs://"):]
    bucket_name, _, blob_name = path.partition("/")
    if not bucket_name or not blob_name:
        raise InputError(f"Malformed GCS URI {uri!r}; expected gs://bucket/path/to/audio.wav")

    if _GCS_CLIENT is None:
        try:
            _GCS_CLIENT = storage.Client()
        except Exception as e:
            raise InputError(f"Could not initialize a GCS client: {type(e).__name__}: {e}") from e

    blob = _GCS_CLIENT.bucket(bucket_name).blob(blob_name)
    try:
        blob.reload()  # populates .size and surfaces 403/404 before we stream
    except Exception as e:
        raise InputError(f"Could not read {uri}: {type(e).__name__}: {e}") from e
    if blob.size and blob.size > MAX_DOWNLOAD_BYTES:
        raise InputError(f"{uri} is {blob.size} bytes, over the {MAX_DOWNLOAD_BYTES} limit.")

    try:
        blob.download_to_filename(dest)
    except Exception as e:
        raise InputError(f"Download of {uri} failed: {type(e).__name__}: {e}") from e

    size = os.path.getsize(dest)
    if size == 0:
        raise InputError(f"{uri} is empty.")
    if size > MAX_DOWNLOAD_BYTES:
        raise InputError(f"{uri} is {size} bytes, over the {MAX_DOWNLOAD_BYTES} limit.")
    return size


def _write_base64(payload: str, dest: str) -> int:
    payload = re.sub(r"\s+", "", payload)
    payload += "=" * (-len(payload) % 4)  # tolerate unpadded base64
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise InputError(f"Input is not valid base64: {e}") from e
    if not raw:
        raise InputError("Decoded base64 audio is empty.")
    if len(raw) > MAX_DOWNLOAD_BYTES:
        raise InputError(f"Decoded audio is {len(raw)} bytes, over the {MAX_DOWNLOAD_BYTES} limit.")
    with open(dest, "wb") as f:
        f.write(raw)
    return len(raw)


def _probe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True, text=True, timeout=120, check=True,
        )
        return float(json.loads(out.stdout)["format"]["duration"])
    except (subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _transcode(src: str, dest: str) -> None:
    """Force to 16 kHz / mono / 16-bit PCM WAV.

    Normalizing through ffmpeg rather than trusting the caller means mp3, m4a,
    ogg, opus, webm and video containers all work.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", src,
            "-vn", "-map", "0:a:0",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le",
            dest,
        ],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(dest) or os.path.getsize(dest) == 0:
        stderr = (proc.stderr or "").strip()[-500:]
        raise InputError(f"ffmpeg could not decode the audio: {stderr or '<no output>'}")


class GpuBusy(RuntimeError):
    """No GPU slot came free in time.

    Its own type because it must never be mistaken for a bad item. The batched
    pass falls back to one clip at a time when a group fails, on the assumption
    that one malformed item poisoned it; doing that after a QUEUE timeout would
    wait the full timeout again for each of up to 32 clips, turning one slow
    request into one that cannot finish inside any sane deadline.
    """


@contextlib.contextmanager
def _gpu_slot():
    """Hold a GPU slot for one forward pass.

    Deliberately narrow. Everything outside it (downloads, ffmpeg, ffprobe,
    decoding the output) runs while another request owns the GPU, which is what
    keeps the device fed instead of waiting on somebody's network.
    """
    if not _gpu_gate.acquire(timeout=GPU_QUEUE_TIMEOUT_S):
        raise GpuBusy(
            f"GPU busy: no slot came free within {GPU_QUEUE_TIMEOUT_S:.0f}s.")
    try:
        yield
    finally:
        _gpu_gate.release()


def _get_fetch_pool() -> "ThreadPoolExecutor":
    """One pool for the whole process, not one per request.

    FETCH_WORKERS is a total, so three concurrent requests share eight
    downloads rather than starting eight each and thrashing eight vCPUs with
    twenty-four ffmpeg processes. Sharing it is also what gives prefetch for
    free: a request's workers are released the moment its items are
    materialised, so the next request starts downloading while the first is
    still on the GPU.

    Never shut down. It lives as long as the process, and a pool that outlives
    every request has nothing to clean up.
    """
    global _fetch_pool
    with _fetch_pool_lock:
        if _fetch_pool is None:
            _fetch_pool = ThreadPoolExecutor(
                max_workers=FETCH_WORKERS, thread_name_prefix="qwen-fetch")
        return _fetch_pool


def _materialize_audio(value: str, workdir: str, idx: int = 0) -> Tuple[str, float, str, int]:
    """Accept a URL, gs:// URI, data URI, or raw base64.

    Returns (wav_path, duration, kind, bytes_in).
    """
    raw = os.path.join(workdir, f"input_{idx}.bin")
    wav = os.path.join(workdir, f"audio_{idx}_16k.wav")

    if value.startswith(("http://", "https://")):
        nbytes = _download(value, raw)
        kind = "url"
    elif value.startswith("gs://"):
        nbytes = _download_gcs(value, raw)
        kind = "gcs"
    elif value.startswith("data:"):
        if ";base64" not in value.split(",", 1)[0]:
            raise InputError("Only base64-encoded data: URIs are supported.")
        nbytes = _write_base64(value.split(",", 1)[1], raw)
        kind = "data-uri"
    elif value.startswith(("file://", "/")):
        # '/' is in the base64 alphabet, so without this branch a path would
        # silently decode to garbage and fail deep inside ffmpeg.
        raise InputError("Local file paths are not accepted. Pass an http(s) URL, a gs:// URI, or base64 audio.")
    else:
        nbytes = _write_base64(value, raw)
        kind = "base64"

    _transcode(raw, wav)
    duration = _probe_duration(wav)
    if duration <= 0:
        raise InputError("Audio contains no decodable samples.")
    if duration > MAX_AUDIO_DURATION_S:
        raise InputError(f"Audio is {duration:.0f}s, over the {MAX_AUDIO_DURATION_S:.0f}s limit.")

    if os.path.exists(raw):
        try:
            os.unlink(raw)
        except OSError:
            pass
    return wav, duration, kind, nbytes


def _split_chunks(path: str, workdir: str) -> List[Tuple[str, float]]:
    """Cut a WAV into overlapping windows: [(path, start_offset_s), ...]."""
    duration = _probe_duration(path)
    stride = max(1.0, CHUNK_LENGTH_S - CHUNK_OVERLAP_S)
    chunks: List[Tuple[str, float]] = []
    start, idx = 0.0, 0
    while start < duration:
        out = os.path.join(workdir, f"chunk_{idx:04d}.wav")
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.3f}", "-t", f"{CHUNK_LENGTH_S:.3f}", "-i", path,
                "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le", out,
            ],
            capture_output=True, text=True, timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) < 1024:
            break
        chunks.append((out, start))
        idx += 1
        start += stride
    if not chunks:
        raise InputError("Chunking produced no usable audio segments.")
    return chunks


# --------------------------------------------------------------------------- #
#                              Transcription                                   #
# --------------------------------------------------------------------------- #
def _transcribe_batch(
    paths: List[str],
    language: Optional[str],
    prompt: Optional[str],
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    """Run the ASR model over a list of WAV paths. Returns [{language, text}, ...]."""
    import torch

    model, processor = _get_model()

    kwargs: Dict[str, Any] = {"audio": paths}
    if language:
        # The processor accepts a per-item list; one hint applies to all here.
        kwargs["language"] = [language] * len(paths) if len(paths) > 1 else language
    if prompt:
        kwargs["prompt"] = prompt

    inputs = processor.apply_transcription_request(**kwargs).to(model.device, model.dtype)

    with _gpu_slot():
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]

    parsed = processor.decode(generated, return_format="parsed")
    out: List[Dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, dict):
            out.append({
                "language": item.get("language"),
                "text": (item.get("transcription") or "").strip(),
            })
        else:  # defensive: older/newer processors may return plain strings
            out.append({"language": None, "text": str(item).strip()})
    return out


def _align_words(wav: str, transcript: str, language: Optional[str],
                 offset_s: float = 0.0) -> List[Dict[str, Any]]:
    """Word-level timestamps via the forced aligner. Never fatal."""
    import torch

    if not transcript.strip():
        return []
    aligner, aligner_processor = _get_aligner()
    try:
        aligner_inputs, word_lists = aligner_processor.prepare_forced_aligner_inputs(
            audio=wav, transcript=transcript, language=language or "English",
        )
        aligner_inputs = aligner_inputs.to(aligner.device, aligner.dtype)
        with _gpu_slot():
            with torch.inference_mode():
                outputs = aligner(**aligner_inputs)
        aligned = aligner_processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=aligner_inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=aligner.config.timestamp_token_id,
        )[0]
    except Exception as e:
        # Alignment is a bonus on top of the transcript; losing it must not
        # cost the caller their transcription.
        print(f"[align] failed ({type(e).__name__}: {e}); returning text without timestamps")
        return []

    words = []
    for item in aligned:
        try:
            words.append({
                "start": round(float(item["start_time"]) + offset_s, 3),
                "end": round(float(item["end_time"]) + offset_s, 3),
                "text": str(item["text"]).strip(),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return words


def _words_to_segments(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group word timings into subtitle-sized segments.

    The aligner emits words only, but SRT/VTT need phrase-level cues. Break on
    sentence punctuation, long pauses, or when a cue gets too long to read.
    """
    segments: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []

    def flush():
        if not cur:
            return
        segments.append({
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
            "text": " ".join(w["text"] for w in cur if w["text"]).strip(),
        })
        cur.clear()

    for i, w in enumerate(words):
        cur.append(w)
        text_len = sum(len(x["text"]) + 1 for x in cur)
        gap_next = (words[i + 1]["start"] - w["end"]) if i + 1 < len(words) else 0.0
        span = w["end"] - cur[0]["start"]

        token = w["text"].lower()
        sentence_end = (
            w["text"].endswith(_SENTENCE_END)
            and token not in _ABBREVIATIONS
            # A one- or two-word "sentence" is nearly always an abbreviation or
            # a stray period, not a real cue boundary.
            and len(cur) >= SEGMENT_MIN_WORDS
        )
        if (sentence_end
                or gap_next > SEGMENT_MAX_GAP_S
                or text_len >= SEGMENT_MAX_CHARS
                or span >= SEGMENT_MAX_DURATION_S):
            flush()
    flush()
    return [s for s in segments if s["text"]]


def _merge_chunks(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stitch overlapping windows back into one transcript.

    Dedup rule: drop any word whose start falls inside the previous window's
    already-accepted span, which is where the overlap duplicates live.
    """
    words: List[Dict[str, Any]] = []
    texts: List[str] = []
    language = None
    accepted_until = 0.0

    for p in parts:
        texts.append(p["text"])
        language = language or p.get("language")
        pw = p.get("words") or []
        for w in pw:
            if w["start"] < accepted_until - 1e-3:
                continue
            words.append(w)
        if pw:
            accepted_until = max(accepted_until, pw[-1]["end"] - CHUNK_OVERLAP_S / 2.0)

    # Prefer text rebuilt from deduplicated words — plain concatenation would
    # repeat whatever the overlap window captured twice.
    text = " ".join(w["text"] for w in words).strip() if words else \
        " ".join(t for t in texts if t).strip()
    return {"text": text, "language": language, "words": words}


# --------------------------------------------------------------------------- #
#                                 Formatting                                   #
# --------------------------------------------------------------------------- #
def _clock(seconds: float, comma: bool) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:  # rounding can push us to a full second
        ms, s = 0, s + 1
    return f"{h:02d}:{m:02d}:{s:02d}{',' if comma else '.'}{ms:03d}"


def _to_srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        if not seg["text"]:
            continue
        lines += [str(i), f"{_clock(seg['start'], True)} --> {_clock(seg['end'], True)}", seg["text"], ""]
    return ("\n".join(lines).strip() + "\n") if lines else ""


def _to_vtt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for seg in segments:
        if not seg["text"]:
            continue
        lines += [f"{_clock(seg['start'], False)} --> {_clock(seg['end'], False)}", seg["text"], ""]
    return "\n".join(lines).strip() + "\n"


def _shape(result: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    words = result.get("words") or []
    segments = result.get("segments") or []

    if fmt == "text":
        return {"text": result.get("text", ""), "language": result.get("language")}
    if fmt == "srt":
        return {"text": result.get("text", ""), "language": result.get("language"), "srt": _to_srt(segments)}
    if fmt == "vtt":
        return {"text": result.get("text", ""), "language": result.get("language"), "vtt": _to_vtt(segments)}
    if fmt == "verbose_json":
        return {
            "text": result.get("text", ""),
            "language": result.get("language"),
            "segments": segments,
            "words": words,
            "duration": result.get("duration"),
        }
    out: Dict[str, Any] = {"text": result.get("text", ""), "language": result.get("language")}
    if words or segments:
        out["timestamps"] = {"word": words, "segment": segments}
    if "duration" in result:
        out["duration"] = result["duration"]
    return out


# --------------------------------------------------------------------------- #
#                                 Validation                                   #
# --------------------------------------------------------------------------- #
def _extract_source(item: Dict[str, Any], where: str) -> str:
    present = [k for k in AUDIO_KEYS if item.get(k)]
    if not present:
        raise InputError(
            f"{where}: provide one of {list(AUDIO_KEYS)} "
            "(an http(s) URL, a gs:// URI, a data: URI, or base64 audio)."
        )
    if len(present) > 1:
        raise InputError(f"{where}: got multiple audio sources {present}; supply exactly one.")
    value = item[present[0]]
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}: '{present[0]}' must be a non-empty string.")
    return value.strip()


def _parse_input(inp: Dict[str, Any]) -> Dict[str, Any]:
    raw_items = inp.get("audio")
    if isinstance(raw_items, list):
        sources = []
        for i, entry in enumerate(raw_items):
            if isinstance(entry, str):
                entry = {"audio": entry}
            if not isinstance(entry, dict):
                raise InputError(f"audio[{i}] must be a string or an object.")
            sources.append(_extract_source(entry, f"audio[{i}]"))
    else:
        sources = [_extract_source(inp, "input")]

    if len(sources) > MAX_BATCH_ITEMS:
        raise InputError(f"Batch of {len(sources)} exceeds the limit of {MAX_BATCH_ITEMS}.")

    fmt = str(inp.get("response_format") or "json").lower()
    if fmt not in VALID_FORMATS:
        raise InputError(f"Unknown response_format {fmt!r}; valid: {sorted(VALID_FORMATS)}")

    timestamps = bool(inp.get("timestamps", False))
    if fmt in {"srt", "vtt", "verbose_json"}:
        timestamps = True  # subtitles are meaningless without timings

    language = inp.get("language")
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise InputError("'language' must be a non-empty string (e.g. \"English\" or \"en\"), or omitted.")

    prompt = inp.get("prompt") or inp.get("context") or inp.get("hotwords")
    if prompt is not None:
        if isinstance(prompt, list):
            prompt = "Vocabulary: " + ", ".join(str(p) for p in prompt) + "."
        if not isinstance(prompt, str):
            raise InputError("'prompt' must be a string or a list of strings.")
        prompt = prompt.strip() or None

    try:
        max_new_tokens = max(16, int(inp.get("max_new_tokens", MAX_NEW_TOKENS)))
    except (TypeError, ValueError):
        raise InputError("'max_new_tokens' must be an integer.") from None

    return {
        "sources": sources,
        "language": language.strip() if isinstance(language, str) else None,
        "prompt": prompt,
        "timestamps": timestamps,
        "response_format": fmt,
        "max_new_tokens": max_new_tokens,
    }


# --------------------------------------------------------------------------- #
#                                   Entry                                      #
# --------------------------------------------------------------------------- #
def _transcribe_one(source: str, req: Dict[str, Any], idx: int) -> Dict[str, Any]:
    workdir = tempfile.mkdtemp(prefix="qwen-asr-", dir=WORK_ROOT)
    try:
        t0 = time.perf_counter()
        wav, duration, kind, nbytes = _materialize_audio(source, workdir, idx)
        fetch_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        if duration > CHUNK_THRESHOLD_S:
            chunks = _split_chunks(wav, workdir)
            print(f"[chunk] {duration:.0f}s split into {len(chunks)} windows")
            parts = []
            for chunk_path, offset in chunks:
                res = _transcribe_batch([chunk_path], req["language"], req["prompt"],
                                        req["max_new_tokens"])[0]
                if req["timestamps"]:
                    res["words"] = _align_words(chunk_path, res["text"], res["language"], offset)
                parts.append(res)
                try:
                    os.unlink(chunk_path)
                except OSError:
                    pass
            result = _merge_chunks(parts)
            strategy = "chunked"
        else:
            result = _transcribe_batch([wav], req["language"], req["prompt"],
                                       req["max_new_tokens"])[0]
            if req["timestamps"]:
                result["words"] = _align_words(wav, result["text"], result["language"])
            strategy = "single_pass"
        infer_s = time.perf_counter() - t1

        result["segments"] = _words_to_segments(result.get("words") or [])
        result["duration"] = round(duration, 3)

        out = _shape(result, req["response_format"])
        out["metrics"] = {
            "audio_seconds": round(duration, 3),
            "fetch_seconds": round(fetch_s, 3),
            "inference_seconds": round(infer_s, 3),
            "realtime_factor": round(duration / infer_s, 1) if infer_s > 0 else None,
            "strategy": strategy,
            "source_kind": kind,
            "input_bytes": nbytes,
            "prompt_used": bool(req["prompt"]),
            "language_forced": bool(req["language"]),
        }
        return out
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _finish_item(res: Dict[str, Any], duration: float, kind: str, nbytes: int,
                 fetch_s: float, infer_s: float, strategy: str,
                 req: Dict[str, Any]) -> Dict[str, Any]:
    """Common tail: segments, duration, output shaping, metrics."""
    res["segments"] = _words_to_segments(res.get("words") or [])
    res["duration"] = round(duration, 3)
    out = _shape(res, req["response_format"])
    out["metrics"] = {
        "audio_seconds": round(duration, 3),
        "fetch_seconds": round(fetch_s, 3),
        "inference_seconds": round(infer_s, 3),
        "realtime_factor": round(duration / infer_s, 1) if infer_s > 0 else None,
        "strategy": strategy,
        "source_kind": kind,
        "input_bytes": nbytes,
        "prompt_used": bool(req["prompt"]),
        "language_forced": bool(req["language"]),
    }
    return out


def _transcribe_grouped(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Transcribe many inputs, sharing GPU forward passes between them.

    Short clips are fetched, sorted by duration, then pushed through the model
    in groups of GPU_BATCH_SIZE. Sorting matters: the processor pads every item
    in a group to the longest one, so mixing a 5s and a 300s clip would waste
    most of the batch on padding.

    Long audio still goes one at a time — it is windowed internally and already
    keeps the GPU busy.
    """
    workdir = tempfile.mkdtemp(prefix="qwen-asr-batch-", dir=WORK_ROOT)
    n = len(req["sources"])
    results: List[Optional[Dict[str, Any]]] = [None] * n
    ready: List[Dict[str, Any]] = []

    try:
        # ---- fetch + normalize everything first, in parallel ----
        # Serially this was the dominant cost of a large batch: 32 downloads,
        # transcodes and duration probes, one after another, with the GPU idle
        # throughout. Each item is independent and its failure is recorded
        # against its own index, so nothing here depends on the order they
        # finish in.
        def _prepare(i: int, source: str):
            t0 = time.perf_counter()
            wav, duration, kind, nbytes = _materialize_audio(source, workdir, i)
            return {"idx": i, "wav": wav, "duration": duration, "kind": kind,
                    "nbytes": nbytes, "fetch_s": time.perf_counter() - t0}

        prepared: Dict[int, Dict[str, Any]] = {}
        pool = _get_fetch_pool()
        futures = {
            pool.submit(_prepare, i, source): i
            for i, source in enumerate(req["sources"])
        }
        for future, i in futures.items():
            try:
                prepared[i] = future.result()
            except InputError as e:
                results[i] = {"error": str(e), "index": i}
            except Exception as e:
                print(f"[error] item {i}: {type(e).__name__}: {e}")
                results[i] = {"error": f"{type(e).__name__}: {e}", "index": i}

        # Rebuilt in the order asked, so a log or a tie in the duration sort
        # below reads the same way it did when this ran serially.
        ready = [prepared[i] for i in sorted(prepared)]

        short = [r for r in ready if r["duration"] <= CHUNK_THRESHOLD_S]
        long_ = [r for r in ready if r["duration"] > CHUNK_THRESHOLD_S]

        # ---- short clips: real GPU batching ----
        short.sort(key=lambda r: r["duration"])
        for start in range(0, len(short), GPU_BATCH_SIZE):
            group = short[start:start + GPU_BATCH_SIZE]
            t1 = time.perf_counter()
            busy: Optional[GpuBusy] = None
            try:
                outs = _transcribe_batch([g["wav"] for g in group], req["language"],
                                         req["prompt"], req["max_new_tokens"])
            except GpuBusy as e:
                # Nothing is wrong with these clips, the device is simply full.
                # Retrying them one at a time would wait the queue timeout 32
                # more times; report the group and move on.
                print(f"[batch] group gave up waiting for the GPU: {e}")
                busy, outs = e, None
            except Exception as e:
                print(f"[batch] group failed ({type(e).__name__}: {e}); retrying individually")
                outs = None
            if busy is not None:
                outs = [{"language": None, "text": "", "_err": f"GpuBusy: {busy}"}
                        for _ in group]
            elif outs is None or len(outs) != len(group):
                # A batched pass can fail on a single malformed item. Falling back
                # one-by-one costs speed but saves the whole group's results.
                outs = []
                for g in group:
                    try:
                        outs.append(_transcribe_batch([g["wav"]], req["language"],
                                                      req["prompt"], req["max_new_tokens"])[0])
                    except Exception as e:
                        outs.append({"language": None, "text": "", "_err": f"{type(e).__name__}: {e}"})
            # Charge each item an equal share of the group's GPU time.
            per_item = (time.perf_counter() - t1) / max(1, len(group))

            for g, res in zip(group, outs):
                if res.get("_err"):
                    results[g["idx"]] = {"error": res["_err"], "index": g["idx"]}
                    continue
                if req["timestamps"]:
                    res["words"] = _align_words(g["wav"], res["text"], res["language"])
                results[g["idx"]] = _finish_item(
                    res, g["duration"], g["kind"], g["nbytes"], g["fetch_s"],
                    per_item, f"gpu_batch_{len(group)}", req)

        # ---- long audio: windowed, one at a time ----
        for g in long_:
            t1 = time.perf_counter()
            try:
                chunks = _split_chunks(g["wav"], workdir)
                parts = []
                for chunk_path, offset in chunks:
                    r = _transcribe_batch([chunk_path], req["language"], req["prompt"],
                                          req["max_new_tokens"])[0]
                    if req["timestamps"]:
                        r["words"] = _align_words(chunk_path, r["text"], r["language"], offset)
                    parts.append(r)
                    try:
                        os.unlink(chunk_path)
                    except OSError:
                        pass
                res = _merge_chunks(parts)
                results[g["idx"]] = _finish_item(
                    res, g["duration"], g["kind"], g["nbytes"], g["fetch_s"],
                    time.perf_counter() - t1, "chunked", req)
            except Exception as e:
                print(f"[error] item {g['idx']}: {type(e).__name__}: {e}")
                results[g["idx"]] = {"error": f"{type(e).__name__}: {e}", "index": g["idx"]}

        # Defensive: nothing may come back as None, or the caller sees a null.
        for i in range(n):
            if results[i] is None:
                results[i] = {"error": "internal: item produced no result", "index": i}
        return results  # type: ignore[return-value]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def transcribe(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Transcribe one or more audio inputs.

    Input schema:
        audio_url (str): http(s) URL of the audio to transcribe.
        gcs_uri (str): gs://bucket/object, read with the runtime service account.
        audio_base64 (str): Base64 audio (padded or not).
        audio (str | list): Base64 string, or a list of items for batch mode.
            Supply exactly one of the above.
        prompt (str | list): Context / hotwords to bias recognition toward,
            e.g. "Vocabulary: Quilter, apostle, gospel." A list is joined into
            that form automatically. Aliases: "context", "hotwords".
        language (str): Force the language, e.g. "English" or "en". Omit to
            auto-detect (the detected value is returned either way).
        timestamps (bool, default False): Word + segment timings via the
            forced aligner.
        response_format (str, default "json"): "json", "verbose_json", "text",
            "srt", or "vtt". The last three force timestamps on.
        max_new_tokens (int, default 448): Generation cap per audio window.

    Returns for a single item:
        {"text": ..., "language": ..., "timestamps": {...}, "duration": float,
         "metrics": {...}, "model": {...}}

    Returns for a batch:
        {"results": [...], "count": int, "failed": int, "model": {...}}

    Returns {"error": str} for bad input or a failed model load. Never raises
    for caller error — the transport layer decides the status code.
    """
    started = time.perf_counter()
    inp = inp or {}

    try:
        req = _parse_input(inp)
    except InputError as e:
        return {"error": str(e)}

    try:
        _get_model()
    except Exception as e:
        return {"error": f"Model failed to load: {type(e).__name__}: {e}"}

    results: List[Dict[str, Any]] = []
    if len(req["sources"]) > 1 and GPU_BATCH_SIZE > 1:
        # Multiple inputs: share GPU forward passes between them.
        results = _transcribe_grouped(req)
    else:
        for i, source in enumerate(req["sources"]):
            try:
                results.append(_transcribe_one(source, req, i))
            except InputError as e:
                results.append({"error": str(e), "index": i})
            except Exception as e:
                # One bad item must not kill the rest of the batch.
                print(f"[error] item {i}: {type(e).__name__}: {e}")
                results.append({"error": f"{type(e).__name__}: {e}", "index": i})

    total_s = round(time.perf_counter() - started, 3)

    if len(results) == 1:
        out = dict(results[0])
        if "error" in out:
            return {"error": out["error"]}
        out["model"] = model_info()
        out["total_seconds"] = total_s
        return out

    return {
        "results": results,
        "count": len(results),
        "failed": sum(1 for r in results if "error" in r),
        "model": model_info(),
        "total_seconds": total_s,
    }
