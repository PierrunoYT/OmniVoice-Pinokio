#!/usr/bin/env python3
"""
OmniVoice HuggingFace Space - Minimal app entrypoint.
Rewritten from scratch for clarity.
"""

import os
import gradio as gr
import torch
import numpy as np

# Try to import spaces (ZeroGPU), stub out if missing for local development.
try:
    import spaces
except ImportError:
    class _GPU:
        def __init__(self, duration=60): pass
        def __call__(self, fn): return fn
    class _Spaces: GPU = _GPU
    spaces = _Spaces()

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo

LANGUAGES = [
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
TAG_CHOICES = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]",
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", "[dissatisfaction-hnn]",
]

def resolve_device(env_var=None):
    if env_var:
        return env_var
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_dtype(device):
    return torch.float16 if device == "cuda" else torch.float32

def env_bool(name, default=True):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ["false", "0", "no", "off"]

def env_int(*names):
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip() != "":
            return int(str(raw).strip())
    return None

# Model loading
ckpt = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
device = resolve_device(os.environ.get("OMNIVOICE_DEVICE"))
dtype = resolve_dtype(device)
load_asr = env_bool("OMNIVOICE_LOAD_ASR", True)

print(f"Loading {ckpt} on {device} ...")
model = OmniVoice.from_pretrained(ckpt, device_map=device, dtype=dtype, load_asr=load_asr)
sampling_rate = model.sampling_rate
print("Model ready.")

def synthesize(text, language, ref_audio, instruct, num_step, guidance, denoise, speed, duration, preproc, postproc, mode, ref_text=None):
    if not text or not str(text).strip():
        return None, "Input text required."
    gen_conf = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance or 2.0),
        denoise=bool(denoise),
        preprocess_prompt=bool(preproc),
        postprocess_output=bool(postproc),
    )
    lang = language if language and language != "Auto" else None
    args = dict(text=str(text).strip(), language=lang, generation_config=gen_conf)
    if speed and float(speed) != 1.0:
        args["speed"] = float(speed)
    if duration and float(duration) > 0:
        args["duration"] = float(duration)
    if mode == "clone":
        if not ref_audio:
            return None, "Reference audio required for cloning."
        args["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design" and instruct and str(instruct).strip():
        args["instruct"] = instruct.strip()
    try:
        audio = model.generate(**args)
    except Exception as e:
        return None, f"Generation error: {type(e).__name__}: {e}"
    arr = audio[0].squeeze(0)
    if hasattr(arr, "detach"): arr = arr.detach().cpu()
    wav = np.clip(arr.numpy(), -1.0, 1.0)
    return (sampling_rate, (wav * 32767).astype(np.int16)), "Done."

import re
def parse_dialogue(script):
    PATTERN = re.compile(r"^\s*\[speaker_(\d+)\]:\s*(.*)$", re.I)
    results = []
    cur_speaker = None
    cur_lines = []
    for line in str(script).strip().splitlines():
        m = PATTERN.match(line)
        if m:
            if cur_speaker is not None and cur_lines:
                results.append((cur_speaker, " ".join(cur_lines).strip()))
            cur_speaker = int(m[1])
            cur_lines = [m[2].strip()] if m[2].strip() else []
            continue
        line = line.strip()
        if line and cur_speaker is not None:
            cur_lines.append(line)
    if cur_speaker and cur_lines:
        results.append((cur_speaker, " ".join(cur_lines).strip()))
    return results

def synthesize_dialogue(
    script, language, num_speakers, num_step, guidance, denoise, speed, duration,
    pause, preprocess, postprocess,
    s1_audio, s1_ref, s1_instr, s1_lang,
    s2_audio, s2_ref, s2_instr, s2_lang,
    s3_audio, s3_ref, s3_instr, s3_lang,
    s4_audio, s4_ref, s4_instr, s4_lang,
):
    if not script or not str(script).strip():
        return None, "Input dialogue required."
    turns = parse_dialogue(script)
    if not turns:
        return None, "No lines found. Use format [Speaker_N]: line"
    n = int(num_speakers or 2)
    speakers = {
        1: {"audio": s1_audio, "ref": s1_ref, "instr": s1_instr, "lang": s1_lang},
        2: {"audio": s2_audio, "ref": s2_ref, "instr": s2_instr, "lang": s2_lang},
        3: {"audio": s3_audio, "ref": s3_ref, "instr": s3_instr, "lang": s3_lang},
        4: {"audio": s4_audio, "ref": s4_ref, "instr": s4_instr, "lang": s4_lang},
    }
    global_lang = language if language and language != "Auto" else None
    gen_conf = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance or 2.0),
        denoise=bool(denoise),
        preprocess_prompt=bool(preprocess),
        postprocess_output=bool(postprocess),
    )
    prompts, audios = {}, []
    for idx, (spk, line) in enumerate(turns, 1):
        if spk not in range(1, n+1):
            return None, f"Line {idx}: Speaker {spk} not in 1..{n}"
        cfg = speakers.get(spk, {})
        lang = cfg.get("lang")
        turn_lang = lang if lang and lang != "Auto" else global_lang
        kw = dict(text=line, language=turn_lang, generation_config=gen_conf)
        if speed and float(speed) != 1.0: kw["speed"] = float(speed)
        if duration and float(duration) > 0: kw["duration"] = float(duration)
        ref_audio = cfg.get("audio")
        ref_text = (cfg.get("ref") or "").strip()
        instr = (cfg.get("instr") or "").strip()
        if ref_audio:
            if spk not in prompts:
                prompts[spk] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text or None)
            kw["voice_clone_prompt"] = prompts[spk]
        if instr: kw["instruct"] = instr
        try:
            result = model.generate(**kw)
        except Exception as e:
            return None, f"Speaker {spk}, line {idx}: {type(e).__name__}: {e}"
        arr = result[0].squeeze(0)
        if hasattr(arr, "detach"): arr = arr.detach().cpu()
        audios.append(arr.numpy().astype(np.float32))
    if not audios: return None, "No output."
    if pause and float(pause) > 0:
        silence = np.zeros(int(float(pause) * sampling_rate), dtype=np.float32)
        merged = audios[0]
        for seg in audios[1:]: merged = np.concatenate([merged, silence, seg], 0)
    else:
        merged = np.concatenate(audios, 0)
    waveform = np.clip(merged, -1, 1)
    return (sampling_rate, (waveform * 32767).astype(np.int16)), f"Done. {len(turns)} lines."

def speaker_box_visibility(num_speakers):
    n = int(num_speakers or 2)
    return [gr.update(visible=(i <= n)) for i in range(1, 5)]

def append_tag_to_text(text, tag):
    t = text or ""
    if not tag: return t, None
    if not t: return tag, None
    sep = "" if t.endswith((" ", "\n")) else " "
    return f"{t}{sep}{tag}", None

def tag_insert_js():
    return """
(selectedTag) => {
  if (!selectedTag) return null;
  const isEditableTextInput = (el) => {
    if (!el) return false;
    const isTextarea = el.tagName === "TEXTAREA";
    const isTextInput = el.tagName === "INPUT" && el.type === "text";
    return (isTextarea || isTextInput) && !el.readOnly && !el.disabled;
  };
  if (!window.__omvTagFocusListener) {
    window.__omvTagFocusListener = true;
    window.__omvLastTextInput = null;
    document.addEventListener("focusin", (ev) => {
      const el = ev.target;
      if (isEditableTextInput(el)) window.__omvLastTextInput = el;
    });
  }
  const active = document.activeElement;
  const el = isEditableTextInput(active) ? active : window.__omvLastTextInput;
  if (!isEditableTextInput(el)) return null;
  const v = el.value || "";
  const start = el.selectionStart ?? v.length, end = el.selectionEnd ?? v.length;
  const left = v.slice(0, start), right = v.slice(end);
  const needsL = left.length > 0 && !/[\\s\\n]$/.test(left);
  const needsR = right.length > 0 && !/^[\\s\\n]/.test(right);
  const ins = `${needsL ? " " : ""}${selectedTag}${needsR ? " " : ""}`;
  const newVal = left + ins + right;
  const caret = (left + ins).length;
  el.value = newVal; el.selectionStart = caret; el.selectionEnd = caret;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
  el.focus();
  return null;
}
""".strip()

def place_vc_tag_row_js():
    return """
() => {
  const group = document.getElementById("vc_tag_group");
  if (!group) return null;

  const findTargetBlock = () => {
    // Primary target: textbox whose placeholder mentions synthesize.
    const textareas = Array.from(document.querySelectorAll("textarea"));
    const targetTextarea = textareas.find((el) =>
      (el.getAttribute("placeholder") || "").toLowerCase().includes("synthesize")
    );
    if (targetTextarea) {
      return targetTextarea.closest("[data-testid='textbox']") || targetTextarea.closest(".block") || targetTextarea.parentElement;
    }

    // Fallback: any label containing "Text to Synthesize".
    const labels = Array.from(document.querySelectorAll("label, span, p, div"));
    const marker = labels.find((el) =>
      (el.textContent || "").toLowerCase().includes("text to synthesize")
    );
    if (!marker) return null;
    return marker.closest("[data-testid='textbox']") || marker.closest(".block") || marker.parentElement;
  };

  let tries = 0;
  const timer = setInterval(() => {
    const targetBlock = findTargetBlock();
    if (targetBlock && targetBlock.parentElement) {
      targetBlock.insertAdjacentElement("afterend", group);
      clearInterval(timer);
      return;
    }
    tries += 1;
    if (tries >= 30) clearInterval(timer);
  }, 100);

  return null;
}
""".strip()

# --- Gradio/Spaces Wrap ---

@spaces.GPU(duration=60)
def generate_fn(*a, **kw): return synthesize(*a, **kw)

@spaces.GPU(duration=120)
def generate_dialogue_fn(*a, **kw): return synthesize_dialogue(*a, **kw)

# --- Build App UI ---

demo = build_demo(model, ckpt, generate_fn=generate_fn)
with demo:
    with gr.Column(elem_id="vc_tag_group"):
        with gr.Row(elem_id="vc_tag_row"):
            vc_tag = gr.Dropdown(label="Insert Tag", choices=TAG_CHOICES, value=None, allow_custom_value=False, scale=5)
            vc_btn = gr.Button("Insert", scale=1, min_width=80)
        gr.Markdown("Tip: use these to insert non-verbal tags into Text to Synthesize. You can also type tags manually (e.g. `[laughter]`).")
    vc_btn.click(fn=None, inputs=[vc_tag], outputs=[vc_tag], js=tag_insert_js())
    demo.load(fn=None, js=place_vc_tag_row_js())
    with gr.Tabs():
        with gr.Tab("Dialogue"):
            gr.Markdown("Generate multi-speaker dialogue with `[Speaker_N]:` tags and per-speaker voice cloning.")
            script = gr.Textbox(label="Dialogue Script", lines=10, value="[Speaker_1]: Hello, I'm speaker one.\n[Speaker_2]: Hi! I'm speaker two.", placeholder="[Speaker_1]: First line\n[Speaker_2]: Reply...")
            with gr.Row():
                d_tag = gr.Dropdown(label="Insert Tag", choices=TAG_CHOICES, value=None, allow_custom_value=False, scale=5)
                d_btn = gr.Button("Insert", scale=1, min_width=80)
            gr.Markdown("Tip: use these to insert tags in the script field. Manual entry also works (e.g. CMU tokens for English).")
            with gr.Row():
                d_lang = gr.Dropdown(label="Language", choices=LANGUAGES, value="Auto", allow_custom_value=True, info="Pick or type language code.")
                d_nspeak = gr.Slider(minimum=2, maximum=4, step=1, value=2, label="Number of Speakers")
                d_pause = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.3, label="Pause Between Speakers (seconds)")
            with gr.Accordion("Generation Settings", open=False):
                with gr.Row():
                    d_nstep = gr.Slider(minimum=4, maximum=64, step=1, value=32, label="num_step")
                    d_guid = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=2.0, label="guidance_scale")
                    d_speed = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="speed")
                    d_dur = gr.Slider(minimum=0.0, maximum=30.0, step=0.5, value=0.0, label="duration (0 = auto)")
                with gr.Row():
                    d_denoise = gr.Checkbox(value=True, label="denoise")
                    d_pre = gr.Checkbox(value=True, label="preprocess_prompt")
                    d_post = gr.Checkbox(value=True, label="postprocess_output")
            spk_boxes, spk_audio, spk_ref, spk_instr, spk_lang = [], [], [], [], []
            with gr.Row():
                for i in range(1, 5):
                    with gr.Column(visible=i<=2) as box:
                        gr.Markdown(f"**Speaker {i}**")
                        a = gr.Audio(label=f"Speaker {i} Reference Audio (optional)", type="filepath")
                        r = gr.Textbox(label=f"Speaker {i} Reference Text (optional)", lines=2, placeholder="Leave empty for ASR.")
                        ins = gr.Textbox(label=f"Speaker {i} Style Instruction (optional)", lines=2, placeholder="e.g. female, low pitch, accent")
                        lang = gr.Dropdown(label=f"Speaker {i} Language", choices=LANGUAGES, value="Auto", allow_custom_value=True, info="Auto uses global language.")
                        spk_boxes.append(box); spk_audio.append(a); spk_ref.append(r); spk_instr.append(ins); spk_lang.append(lang)
            d_run = gr.Button("Generate Dialogue", variant="primary")
            d_audio = gr.Audio(label="Dialogue Output")
            d_status = gr.Textbox(label="Status", interactive=False)
            d_nspeak.change(fn=speaker_box_visibility, inputs=[d_nspeak], outputs=spk_boxes)
            d_btn.click(append_tag_to_text, inputs=[script, d_tag], outputs=[script, d_tag])
            d_run.click(
                generate_dialogue_fn,
                inputs=[
                    script, d_lang, d_nspeak, d_nstep, d_guid, d_denoise, d_speed, d_dur, d_pause, d_pre, d_post,
                    spk_audio[0], spk_ref[0], spk_instr[0], spk_lang[0],
                    spk_audio[1], spk_ref[1], spk_instr[1], spk_lang[1],
                    spk_audio[2], spk_ref[2], spk_instr[2], spk_lang[2],
                    spk_audio[3], spk_ref[3], spk_instr[3], spk_lang[3],
                ],
                outputs=[d_audio, d_status],
                api_name="generate_dialogue"
            )

if __name__ == "__main__":
    launch_args = {"inbrowser": False}
    host = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
    launch_args["server_name"] = host
    port = env_int("OMNIVOICE_PORT", "PORT", "GRADIO_SERVER_PORT")
    if port: launch_args["server_port"] = port
    demo.queue().launch(**launch_args)
