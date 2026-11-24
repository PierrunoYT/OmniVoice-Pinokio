# OmniVoice (Pinokio)

[OmniVoice](https://github.com/k2-fsa/OmniVoice) zero-shot multilingual TTS (600+ languages) with voice cloning and voice design. Launcher scripts live in the repo root; application code is under `app/`.

## What it does

- Installs **PyTorch** via `torch.js` (per platform), then Python deps from `app/requirements.txt` (including `omnivoice`).
- Validated against **`omnivoice>=0.1.3,<0.2`**, **`gradio>=5.0,<7`** and **`transformers>=5.3,<6`**. These pins live in `app/requirements.txt` — bump them there and re-run **Install** or **Update** when upgrading.
- Starts the **Gradio** UI from **`app/app.py`** (same flow as the [Hugging Face Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)).
- Uses **one** Pinokio virtual environment: **`env/`** at the **project root** (not under `app/`). If you see both `env/` and `app/env/`, remove `app/env/` and rely on **Install** so PyTorch and packages live in the root `env/` only.

## Using in Pinokio

1. **Install** — creates `env/`, runs `torch.js` (platform-specific PyTorch), then `uv pip install -r app/requirements.txt`.
2. **Start** — from the project root runs **`python app/app.py`** with **`OMNIVOICE_PORT={{port}}`** (next free port). Gradio binds to **`127.0.0.1`** by default; see environment variables below.
3. **Update** — `git pull` (if this folder is a git repo), then upgrades the app's direct deps (`omnivoice`, `transformers`, `accelerate`, `gradio`, …) within the `app/requirements.txt` pins and re-runs `torch.js`. PyTorch itself is left as installed by `torch.js`.
4. **Reset** — removes `env/` for a clean reinstall.
5. **Save Disk Space** — `link.js` deduplicates venv library files.

## Environment variables (`app/app.py`)

| Variable | Purpose |
|----------|---------|
| `OMNIVOICE_MODEL` | Hugging Face repo or checkpoint (default `k2-fsa/OmniVoice`). |
| `OMNIVOICE_DEVICE` | Force `cuda`, `mps`, `directml`, or `cpu` (case-insensitive; default: auto cuda → mps → directml → cpu). Unknown or unavailable values fall back to auto-detection with a warning. |
| `OMNIVOICE_LOAD_ASR` | `0` / `false` to skip Whisper ASR (less VRAM; supply reference text for clone). |
| `OMNIVOICE_HOST` | Gradio `server_name` (default **`127.0.0.1`**). Use `0.0.0.0` to listen on all interfaces. |
| `OMNIVOICE_PORT`, `PORT`, `GRADIO_SERVER_PORT` | Gradio port, `1`-`65535` (Pinokio sets `OMNIVOICE_PORT`). Unparseable or out-of-range values are ignored with a warning and Gradio picks the port. |

### Hugging Face downloads (first run)

The first **Start** downloads checkpoints from the Hub (several GB). You may see:

- **`Warning: You are sending unauthenticated requests`** — set a [Hugging Face token](https://huggingface.co/settings/tokens) as **`HF_TOKEN`** in Pinokio (environment / secrets) or in your user environment so Hub rate limits are higher and downloads are faster.
- **Slow transfer** — this launcher sets **`HF_XET_HIGH_PERFORMANCE=1`** in `start.js` (Xet high-performance mode; `hf_transfer` / `HF_HUB_ENABLE_HF_TRANSFER` are no longer used by `huggingface_hub` 1.x).
- **Mirror (e.g. region / connectivity)** — optional: `HF_ENDPOINT` (see [OmniVoice README](https://github.com/k2-fsa/OmniVoice) for `hf-mirror` and similar).
- **Offline** — at startup the app probes the Hub (`HF_ENDPOINT` if set, else `huggingface.co`). If it is unreachable, it sets `HF_HUB_OFFLINE=1` and loads models from the local cache, so previously downloaded models still start without internet. The probe is skipped when `HF_HUB_OFFLINE` is already set or an HTTPS proxy is configured.

### Device (CPU vs GPU)

Logs like **`Loading model ... to cpu`** mean PyTorch selected **CPU** (no usable CUDA/MPS/DirectML, or CPU-only PyTorch). For **NVIDIA** acceleration, install with **Install** so `torch.js` installs the **CUDA** build, and ensure a recent GPU driver. Forcing **`OMNIVOICE_DEVICE=cuda`** (or `mps`) only takes effect when that backend is actually available; otherwise the app warns and falls back to auto-detection rather than failing to load. On **Windows + AMD**, `torch.js` installs **`torch-directml`** and the app auto-detects it (or force **`OMNIVOICE_DEVICE=directml`**).

## Programmatic access

After **Start**, open the URL from Pinokio (“Open Web UI”) or the logs. The server is **`http://127.0.0.1:<port>`** by default; replace `PORT` below with the port shown in Pinokio.

The Dialogue tab is exposed as the **`/generate_dialogue`** endpoint. It takes 27 positional inputs and returns `[audio, status]`:

| # | Input | Example |
|---|-------|---------|
| 1 | Dialogue script (`[Speaker_N]: line`, one turn per line) | `"[Speaker_1]: Hello.\n[Speaker_2]: Hi!"` |
| 2 | Language code or `"Auto"` | `"en"` |
| 3 | Number of speakers (`2`–`4`) | `2` |
| 4–8 | `num_step`, `guidance_scale`, `denoise`, `speed`, `duration` (`0` = auto) | `32, 2.0, true, 1.0, 0.0` |
| 9–11 | Pause between speakers (s), `preprocess_prompt`, `postprocess_output` | `0.3, true, true` |
| 12–27 | Per speaker 1–4: reference audio (file or `null`), reference text, style instruction, language (`"Auto"` = global) | `null, "", "", "Auto"` |

### cURL

```bash
EVENT_ID=$(curl -sS -X POST "http://127.0.0.1:PORT/gradio_api/call/generate_dialogue" \
  -H "Content-Type: application/json" \
  -d '{"data": ["[Speaker_1]: Hello there.\n[Speaker_2]: Hi!", "Auto", 2, 32, 2.0, true, 1.0, 0.0, 0.3, true, true,
       null, "", "", "Auto", null, "", "", "Auto", null, "", "", "Auto", null, "", "", "Auto"]}' \
  | sed -E 's/.*"event_id": *"([^"]+)".*/\1/')
curl -sS -N "http://127.0.0.1:PORT/gradio_api/call/generate_dialogue/$EVENT_ID"
```

The final `event: complete` message contains the output file URL and the status text.

### Python

Uses the [Gradio Python client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`pip install gradio_client`):

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:PORT/")
speaker = [None, "", "", "Auto"]  # no reference audio: model picks a voice
cloned = [handle_file("speaker1.wav"), "", "", "Auto"]  # empty ref text = ASR
audio_path, status = client.predict(
    "[Speaker_1]: Hello there.\n[Speaker_2]: Hi! Nice to meet you.",
    "Auto", 2, 32, 2.0, True, 1.0, 0.0, 0.3, True, True,
    *cloned, *speaker, *speaker, *speaker,
    api_name="/generate_dialogue",
)
print(audio_path, status)
```

### JavaScript

Uses the [Gradio JavaScript client](https://www.gradio.app/guides/getting-started-with-the-js-client) (`npm install @gradio/client`):

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:PORT/");
const speaker = [null, "", "", "Auto"];
const result = await client.predict("/generate_dialogue", [
  "[Speaker_1]: Hello there.\n[Speaker_2]: Hi! Nice to meet you.",
  "Auto", 2, 32, 2.0, true, 1.0, 0.0, 0.3, true, true,
  ...speaker, ...speaker, ...speaker, ...speaker,
]);
const [audio, status] = result.data;
console.log(audio.url, status);
```

The upstream Voice Clone / Voice Design tabs are also callable; run `client.view_api()` (Python) or open `http://127.0.0.1:PORT/?view=api` to list every endpoint and its inputs.

## Project layout

```
project-root/
├── app/
│   ├── app.py
│   └── requirements.txt
├── install.js, start.js, update.js, reset.js, link.js, torch.js
├── pinokio.js, pinokio.json
└── README.md
```

## Citation

OmniVoice: [arXiv:2604.00688](https://arxiv.org/abs/2604.00688), [GitHub](https://github.com/k2-fsa/OmniVoice).
