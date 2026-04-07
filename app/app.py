#!/usr/bin/env python3
"""
HuggingFace Space entry point for OmniVoice demo.

Same structure as the official Space; locally, ``spaces`` is stubbed if missing, and
``OMNIVOICE_*`` env vars configure device and Gradio bind (see ``start.js``).
"""

import os
import re
from typing import Any, Dict

import gradio as gr
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

LANGUAGE_CHOICES = [
    ("Auto", "Auto"),
    ("Arabic (ar)", "ar"),
    ("Chinese (zh)", "zh"),
    ("Dutch (nl)", "nl"),
    ("English (en)", "en"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Hindi (hi)", "hi"),
    ("Italian (it)", "it"),
    ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"),
    ("Polish (pl)", "pl"),
    ("Portuguese (pt)", "pt"),
    ("Russian (ru)", "ru"),
    ("Spanish (es)", "es"),
    ("Turkish (tr)", "tr"),
]

NON_VERBAL_TAG_CHOICES = [
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
]


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


def _parse_dialogue_script(script: str):
    tag_re = re.compile(r"^\s*\[speaker_(\d+)\]:\s*(.*)$", re.IGNORECASE)
    turns = []
    current_speaker = None
    current_parts = []
    for raw in script.strip().splitlines():
        match = tag_re.match(raw)
        if match:
            if current_speaker is not None and current_parts:
                turns.append((current_speaker, " ".join(current_parts).strip()))
            current_speaker = int(match.group(1))
            initial_text = match.group(2).strip()
            current_parts = [initial_text] if initial_text else []
            continue
        stripped = raw.strip()
        if stripped and current_speaker is not None:
            current_parts.append(stripped)
    if current_speaker is not None and current_parts:
        turns.append((current_speaker, " ".join(current_parts).strip()))
    return turns


def _generate_dialogue(
    script,
    language,
    num_speakers,
    num_step,
    guidance_scale,
    denoise,
    speed,
    duration,
    pause_between_speakers,
    preprocess_prompt,
    postprocess_output,
    s1_audio,
    s1_ref_text,
    s1_instruct,
    s1_language,
    s2_audio,
    s2_ref_text,
    s2_instruct,
    s2_language,
    s3_audio,
    s3_ref_text,
    s3_instruct,
    s3_language,
    s4_audio,
    s4_ref_text,
    s4_instruct,
    s4_language,
):
    if not script or not script.strip():
        return None, "Please enter dialogue text."

    turns = _parse_dialogue_script(script)
    if not turns:
        return None, "No valid dialogue lines found. Use [Speaker_N]: text format."

    n = int(num_speakers or 2)
    speaker_data = {
        1: {
            "audio": s1_audio,
            "ref_text": s1_ref_text,
            "instruct": s1_instruct,
            "language": s1_language,
        },
        2: {
            "audio": s2_audio,
            "ref_text": s2_ref_text,
            "instruct": s2_instruct,
            "language": s2_language,
        },
        3: {
            "audio": s3_audio,
            "ref_text": s3_ref_text,
            "instruct": s3_instruct,
            "language": s3_language,
        },
        4: {
            "audio": s4_audio,
            "ref_text": s4_ref_text,
            "instruct": s4_instruct,
            "language": s4_language,
        },
    }

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )
    lang = language if (language and language != "Auto") else None

    prompt_cache = {}
    audio_turns = []
    for idx, (speaker_id, line_text) in enumerate(turns, start=1):
        if speaker_id < 1 or speaker_id > n:
            return (
                None,
                f"Line {idx} uses [Speaker_{speaker_id}], but active speakers are 1..{n}.",
            )

        speaker_cfg = speaker_data.get(speaker_id, {})
        speaker_lang = speaker_cfg.get("language")
        turn_lang = (
            speaker_lang
            if (speaker_lang and speaker_lang != "Auto")
            else lang
        )
        kw: Dict[str, Any] = {
            "text": line_text,
            "language": turn_lang,
            "generation_config": gen_config,
        }
        if speed is not None and float(speed) != 1.0:
            kw["speed"] = float(speed)
        if duration is not None and float(duration) > 0:
            kw["duration"] = float(duration)

        ref_audio = speaker_cfg.get("audio")
        ref_text = (speaker_cfg.get("ref_text") or "").strip()
        instruct = (speaker_cfg.get("instruct") or "").strip()

        if ref_audio:
            if speaker_id not in prompt_cache:
                prompt_cache[speaker_id] = model.create_voice_clone_prompt(
                    ref_audio=ref_audio, ref_text=ref_text or None
                )
            kw["voice_clone_prompt"] = prompt_cache[speaker_id]
        if instruct:
            kw["instruct"] = instruct

        try:
            audio = model.generate(**kw)
        except Exception as e:
            return None, f"Error on speaker {speaker_id}, line {idx}: {type(e).__name__}: {e}"
        t = audio[0].squeeze(0)
        if hasattr(t, "detach"):
            t = t.detach().cpu()
        audio_turns.append(t.numpy().astype(np.float32))

    if not audio_turns:
        return None, "No audio generated."

    if float(pause_between_speakers or 0) > 0:
        silence = np.zeros(
            int(float(pause_between_speakers) * sampling_rate), dtype=np.float32
        )
        merged = audio_turns[0]
        for turn in audio_turns[1:]:
            merged = np.concatenate([merged, silence, turn], axis=0)
    else:
        merged = np.concatenate(audio_turns, axis=0)

    waveform = np.clip(merged, -1.0, 1.0)
    waveform = (waveform * 32767).astype(np.int16)
    return (sampling_rate, waveform), f"Done. Generated {len(turns)} line(s)."


def _speaker_visibility_updates(num_speakers):
    n = int(num_speakers or 2)
    return [gr.update(visible=i <= n) for i in range(1, 5)]


def _append_selected_tag(text_value: str, selected_tag: str):
    current = text_value or ""
    if not selected_tag:
        return current, None
    if not current:
        return selected_tag, None
    sep = "" if current.endswith((" ", "\n")) else " "
    return f"{current}{sep}{selected_tag}", None


def _insert_tag_into_focused_js() -> str:
    return """
(selectedTag) => {
  if (!selectedTag) return null;
  const active = document.activeElement;
  if (!active) return null;
  const isTextarea = active.tagName === "TEXTAREA";
  const isTextInput = active.tagName === "INPUT" && active.type === "text";
  if (!isTextarea && !isTextInput) return null;

  const value = active.value || "";
  const start = active.selectionStart ?? value.length;
  const end = active.selectionEnd ?? value.length;
  const left = value.slice(0, start);
  const right = value.slice(end);
  const needsLeftSpace = left.length > 0 && !/[\\s\\n]$/.test(left);
  const needsRightSpace = right.length > 0 && !/^[\\s\\n]/.test(right);
  const insertion = `${needsLeftSpace ? " " : ""}${selectedTag}${needsRightSpace ? " " : ""}`;
  const nextValue = left + insertion + right;
  const caretPos = (left + insertion).length;

  active.value = nextValue;
  active.selectionStart = caretPos;
  active.selectionEnd = caretPos;
  active.dispatchEvent(new Event("input", { bubbles: true }));
  active.dispatchEvent(new Event("change", { bubbles: true }));
  active.focus();
  return null;
}
""".strip()


# ---------------------------------------------------------------------------
# ZeroGPU wrapper
# ---------------------------------------------------------------------------


@spaces.GPU(duration=60)
def generate_fn(*args, **kwargs):
    return _gen_core(*args, **kwargs)


@spaces.GPU(duration=120)
def generate_dialogue_fn(*args, **kwargs):
    return _generate_dialogue(*args, **kwargs)


# ---------------------------------------------------------------------------
# Build and launch demo
# ---------------------------------------------------------------------------
demo = build_demo(model, CHECKPOINT, generate_fn=generate_fn)
with demo:
    with gr.Row():
        g_tag_picker = gr.Dropdown(
            label="Quick Tag Insert (Single TTS + Dialogue)",
            choices=NON_VERBAL_TAG_CHOICES,
            value=None,
            allow_custom_value=False,
            scale=5,
        )
        g_insert_tag = gr.Button("Insert Into Focused Text Box", scale=2, min_width=180)
    g_insert_tag.click(
        fn=None,
        inputs=[g_tag_picker],
        outputs=[g_tag_picker],
        js=_insert_tag_into_focused_js(),
    )
    with gr.Tabs():
        with gr.Tab("Dialogue"):
            gr.Markdown(
                "Generate multi-speaker dialogue with `[Speaker_N]:` tags and optional per-speaker voice cloning."
            )
            script = gr.Textbox(
                label="Dialogue Script",
                lines=10,
                value="[Speaker_1]: Hello, I'm speaker one.\n[Speaker_2]: Hi! I'm speaker two.",
                placeholder="[Speaker_1]: First line\n[Speaker_2]: Reply...",
            )
            with gr.Row():
                d_tag_picker = gr.Dropdown(
                    label="Insert Tag",
                    choices=NON_VERBAL_TAG_CHOICES,
                    value=None,
                    allow_custom_value=False,
                    scale=5,
                )
                d_insert_tag = gr.Button("Insert", scale=1, min_width=80)
            gr.Markdown(
                "Tip: tag controls are placed next to the Dialogue Script input. You can still type pronunciation controls manually (e.g. CMU tokens for English)."
            )
            with gr.Row():
                d_language = gr.Dropdown(
                    label="Language",
                    choices=LANGUAGE_CHOICES,
                    value="Auto",
                    allow_custom_value=True,
                    info="Pick a language code or type a custom value.",
                )
                d_num_speakers = gr.Slider(
                    minimum=2, maximum=4, step=1, value=2, label="Number of Speakers"
                )
                d_pause = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    step=0.1,
                    value=0.3,
                    label="Pause Between Speakers (seconds)",
                )
            with gr.Accordion("Generation Settings", open=False):
                with gr.Row():
                    d_num_step = gr.Slider(
                        minimum=4, maximum=64, step=1, value=32, label="num_step"
                    )
                    d_guidance = gr.Slider(
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=2.0,
                        label="guidance_scale",
                    )
                    d_speed = gr.Slider(
                        minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="speed"
                    )
                    d_duration = gr.Slider(
                        minimum=0.0,
                        maximum=30.0,
                        step=0.5,
                        value=0.0,
                        label="duration (0 = auto)",
                    )
                with gr.Row():
                    d_denoise = gr.Checkbox(value=True, label="denoise")
                    d_pre = gr.Checkbox(value=True, label="preprocess_prompt")
                    d_post = gr.Checkbox(value=True, label="postprocess_output")
            speaker_boxes = []
            speaker_audio = []
            speaker_ref = []
            speaker_instr = []
            with gr.Row():
                for i in range(1, 5):
                    with gr.Column(visible=i <= 2) as col:
                        gr.Markdown(f"**Speaker {i}**")
                        a = gr.Audio(
                            label=f"Speaker {i} Reference Audio (optional)",
                            type="filepath",
                        )
                        r = gr.Textbox(
                            label=f"Speaker {i} Reference Text (optional)",
                            lines=2,
                            placeholder="Leave empty for auto ASR if enabled.",
                        )
                        ins = gr.Textbox(
                            label=f"Speaker {i} Style Instruction (optional)",
                            lines=2,
                            placeholder="e.g. female, low pitch, british accent",
                        )
                        lang = gr.Dropdown(
                            label=f"Speaker {i} Language",
                            choices=LANGUAGE_CHOICES,
                            value="Auto",
                            allow_custom_value=True,
                            info="Auto uses the global language setting.",
                        )
                        speaker_boxes.append(col)
                        speaker_audio.append(a)
                        speaker_ref.append(r)
                        speaker_instr.append(ins)
                        if i == 1:
                            s1_lang = lang
                        elif i == 2:
                            s2_lang = lang
                        elif i == 3:
                            s3_lang = lang
                        else:
                            s4_lang = lang
            d_run = gr.Button("Generate Dialogue", variant="primary")
            d_audio_out = gr.Audio(label="Dialogue Output")
            d_status = gr.Textbox(label="Status", interactive=False)

            d_num_speakers.change(
                fn=_speaker_visibility_updates,
                inputs=[d_num_speakers],
                outputs=speaker_boxes,
            )
            d_insert_tag.click(
                fn=_append_selected_tag,
                inputs=[script, d_tag_picker],
                outputs=[script, d_tag_picker],
            )
            d_run.click(
                fn=generate_dialogue_fn,
                inputs=[
                    script,
                    d_language,
                    d_num_speakers,
                    d_num_step,
                    d_guidance,
                    d_denoise,
                    d_speed,
                    d_duration,
                    d_pause,
                    d_pre,
                    d_post,
                    speaker_audio[0],
                    speaker_ref[0],
                    speaker_instr[0],
                    s1_lang,
                    speaker_audio[1],
                    speaker_ref[1],
                    speaker_instr[1],
                    s2_lang,
                    speaker_audio[2],
                    speaker_ref[2],
                    speaker_instr[2],
                    s3_lang,
                    speaker_audio[3],
                    speaker_ref[3],
                    speaker_instr[3],
                    s4_lang,
                ],
                outputs=[d_audio_out, d_status],
                api_name="generate_dialogue",
            )

if __name__ == "__main__":
    launch_kw = {"inbrowser": False}
    # Gepeto/Pinokio: prefer localhost; set OMNIVOICE_HOST=0.0.0.0 to listen on all interfaces
    host = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
    launch_kw["server_name"] = host
    port = _env_int("OMNIVOICE_PORT", "PORT", "GRADIO_SERVER_PORT")
    if port is not None:
        launch_kw["server_port"] = port
    demo.queue().launch(**launch_kw)
