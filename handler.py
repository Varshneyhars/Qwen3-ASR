"""RunPod serverless handler for Qwen3-ASR-1.7B.

All transcription logic lives in asr_core so the same code serves both the
RunPod queue worker and the Cloud Run HTTP server (server.py).
"""
import runpod

import asr_core
from asr_core import transcribe


def handler(event):
    """RunPod serverless entry point.

    event["input"] follows the schema documented on asr_core.transcribe.
    """
    return transcribe(event.get("input") or {})


# Re-exported so existing scripts that poked at handler internals keep working.
InputError = asr_core.InputError
MODEL_ID = asr_core.MODEL_ID


# Started at module level, not under `if __name__ == "__main__"`, so RunPod's
# GitHub deploy scan can statically detect the entrypoint for queue-based
# endpoints. The container's only entrypoint is `python -u handler.py`, so this
# is equivalent at runtime. Model loading stays lazy (see asr_core._get_model),
# so this returns control to the SDK immediately and the worker passes
# readiness.
runpod.serverless.start({"handler": handler})
