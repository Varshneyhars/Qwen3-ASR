"""Bulk-transcribe many short files through a Cloud Run Qwen3-ASR endpoint.

Sized for the case this is worst at if done naively: tens of thousands of clips
only seconds long. Sending those one per request costs ~20x more than batching
them, because a 70s cold start and a 1.7s connection setup dwarf a 3s
transcription. This groups files into batches, keeps several in flight so the
service stays warm, and can resume after a failure.

    # 1. list the work
    gcloud storage ls "gs://my-bucket/clips/**.wav" > files.txt

    # 2. run it
    python batch_client.py --service-url https://qwen3-asr-xxxx.run.app \
        --input files.txt --output results.jsonl

Re-running skips anything already in --output, so an interrupted job resumes
where it stopped rather than re-paying for completed work.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional

import requests

# The server caps a request at QWEN_MAX_BATCH_ITEMS (default 32). Asking for
# more is rejected outright, so this is a ceiling and not a tuning knob.
DEFAULT_BATCH_SIZE = 32
# Match the service's --max-instances. Going higher just queues at the load
# balancer; it doesn't buy throughput, and every queued request burns its
# 3600s timeout budget while waiting.
DEFAULT_CONCURRENCY = 3


class TokenCache:
    """gcloud identity tokens last an hour; mint one and reuse it.

    Shelling out per request would add ~300ms to every batch and, at this
    volume, dominate the wall clock.
    """

    def __init__(self, static_token: Optional[str] = None, ttl_s: int = 2700):
        self._static = static_token
        self._ttl = ttl_s
        self._token: Optional[str] = None
        self._minted_at = 0.0
        self._lock = threading.Lock()

    def get(self) -> str:
        if self._static:
            return self._static
        with self._lock:
            if self._token is None or (time.time() - self._minted_at) > self._ttl:
                self._token = subprocess.run(
                    ["gcloud", "auth", "print-identity-token"],
                    capture_output=True, text=True, check=True, shell=(os.name == "nt"),
                ).stdout.strip()
                self._minted_at = time.time()
            return self._token


def read_sources(path: str) -> List[str]:
    """One URI per line: gs://..., https://..., or a local path to base64."""
    out: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def load_done(path: str) -> set:
    """URIs already recorded in the output file, so a resume skips them."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a hard kill
            if rec.get("source"):
                done.add(rec["source"])
    return done


def chunk(items: List[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def key_for(uri: str) -> str:
    return "gcs_uri" if uri.startswith("gs://") else "audio_url"


def send_batch(
    session: requests.Session,
    url: str,
    tokens: TokenCache,
    sources: List[str],
    response_format: str,
    language: Optional[str],
    prompt: Optional[str],
    timeout_s: float,
    retries: int,
) -> List[Dict[str, Any]]:
    """POST one batch, retrying transient failures with backoff."""
    payload: Dict[str, Any] = {
        "audio": [{key_for(s): s} for s in sources],
        "response_format": response_format,
    }
    if language:
        payload["language"] = language
    if prompt:
        payload["prompt"] = prompt

    last_err = ""
    for attempt in range(retries + 1):
        try:
            resp = session.post(
                url.rstrip("/") + "/transcribe",
                json=payload,
                headers={"Authorization": f"Bearer {tokens.get()}"},
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                body = resp.json()
                # A single-item batch comes back unwrapped, not as {"results": [...]}.
                results = body.get("results") or [body]
                return [{**r, "source": s} for s, r in zip(sources, results)]
            # 400 means the payload is wrong; retrying re-pays for the same
            # rejection. Only 429/5xx are worth another attempt.
            if resp.status_code < 500 and resp.status_code != 429:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                break
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"

        if attempt < retries:
            backoff = min(60, 2 ** attempt * 5)
            print(f"  retry {attempt + 1}/{retries} in {backoff}s ({last_err[:120]})",
                  file=sys.stderr)
            time.sleep(backoff)

    return [{"source": s, "error": last_err} for s in sources]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--service-url", required=True, help="https://qwen3-asr-xxxx.run.app")
    ap.add_argument("--input", required=True, help="File with one audio URI per line")
    ap.add_argument("--output", default="results.jsonl", help="JSONL output (append + resume)")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help="In-flight requests; set to the service's --max-instances")
    ap.add_argument("--response-format", default="text",
                    choices=["json", "verbose_json", "text", "srt", "vtt"])
    ap.add_argument("--language", default=None, help='Force e.g. "English" to skip detection')
    ap.add_argument("--prompt", default=None, help="Context biasing / hotwords")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--token", default=None,
                    help="Bearer token; default mints one via gcloud")
    args = ap.parse_args()

    if args.batch_size > 32:
        print("WARNING: server default caps batches at 32; larger requests are rejected.",
              file=sys.stderr)

    sources = read_sources(args.input)
    done = load_done(args.output)
    todo = [s for s in sources if s not in done]

    print(f"{len(sources):,} inputs | {len(done):,} already done | {len(todo):,} to process")
    if not todo:
        print("Nothing to do.")
        return 0

    batches = list(chunk(todo, args.batch_size))
    print(f"{len(batches):,} batches of up to {args.batch_size}, "
          f"{args.concurrency} in flight\n")

    tokens = TokenCache(args.token)
    session = requests.Session()
    write_lock = threading.Lock()
    started = time.perf_counter()
    ok = failed = 0
    audio_s = 0.0

    with open(args.output, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(send_batch, session, args.service_url, tokens, b,
                        args.response_format, args.language, args.prompt,
                        args.timeout, args.retries): i
            for i, b in enumerate(batches)
        }
        for n, fut in enumerate(as_completed(futures), start=1):
            results = fut.result()
            with write_lock:
                for r in results:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    if "error" in r:
                        failed += 1
                    else:
                        ok += 1
                        audio_s += (r.get("metrics") or {}).get("audio_seconds", 0.0)
                out.flush()  # a crash must not lose completed work
            elapsed = time.perf_counter() - started
            print(f"[{n}/{len(batches)}] ok={ok:,} failed={failed:,} "
                  f"| {audio_s / 3600:.2f} h audio | {elapsed / 60:.1f} min elapsed"
                  + (f" | ETA {elapsed / n * (len(batches) - n) / 60:.0f} min"
                     if n < len(batches) else ""))

    elapsed = time.perf_counter() - started
    print(f"\nDone in {elapsed / 60:.1f} min — {ok:,} ok, {failed:,} failed")
    print(f"Processed {audio_s / 3600:.2f} h of audio -> {args.output}")
    if failed:
        print(f"Re-run the same command to retry the {failed:,} failures.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
