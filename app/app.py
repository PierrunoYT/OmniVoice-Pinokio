#!/usr/bin/env python3
"""
HuggingFace Space entry point for OmniVoice demo.

Same structure as the official Space; locally, ``spaces`` is stubbed if missing, and
``OMNIVOICE_*`` env vars configure device and Gradio bind (see ``start.js``).
"""

import os
from typing import Any, Dict

import numpy as np
import torch

try:
    import spaces
except ImportError:

    class _GPU:
        def __init__(self, duration=60):
            pass

        def __call__(self, fn):
            return fn

    class _Spaces:
        GPU = _GPU

    spaces = _Spaces()

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo


def _resolve_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str):
    return torch.float16 if device == "cuda" else torch.float32


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_int(*names: str):
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip() != "":
            return int(str(raw).strip())
    return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

_device = _resolve_device(os.environ.get("OMNIVOICE_DEVICE") or None)
_dtype = _resolve_dtype(_device)
_load_asr = _env_flag("OMNIVOICE_LOAD_ASR", True)

print(f"Loading model from {CHECKPOINT} to {_device} ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map=_device,
    dtype=_dtype,
    load_asr=_load_asr,
)
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def _gen_core(
    text,
    language,
    ref_audio,
    instruct,
    num_step,
    guidance_scale,
    denoise,
    speed,
    duration,
    preprocess_prompt,
    postprocess_output,
    mode,
    ref_text=None,
):
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    kw: Dict[str, Any] = dict(
        text=text.strip(), language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    if mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    t = audio[0].squeeze(0)
    if hasattr(t, "detach"):
        t = t.detach().cpu()
    waveform = t.numpy()
    waveform = (waveform * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."


# ---------------------------------------------------------------------------
# ZeroGPU wrapper
# ---------------------------------------------------------------------------


@spaces.GPU(duration=60)
def generate_fn(*args, **kwargs):
    return _gen_core(*args, **kwargs)


# ---------------------------------------------------------------------------
# Build and launch demo
# ---------------------------------------------------------------------------
demo = build_demo(model, CHECKPOINT, generate_fn=generate_fn)

if __name__ == "__main__":
    launch_kw = {"inbrowser": False}
    # Gepeto/Pinokio: prefer localhost; set OMNIVOICE_HOST=0.0.0.0 to listen on all interfaces
    host = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
    launch_kw["server_name"] = host
    port = _env_int("OMNIVOICE_PORT", "PORT", "GRADIO_SERVER_PORT")
    if port is not None:
        launch_kw["server_port"] = port
    demo.queue().launch(**launch_kw)
