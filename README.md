# OmniVoice (Pinokio)

[OmniVoice](https://github.com/k2-fsa/OmniVoice) zero-shot multilingual TTS (600+ languages) with voice cloning and voice design. Launcher scripts live in the repo root; application code is under `app/`.

## What it does

- Installs **PyTorch** via `torch.js` (per platform), then Python deps from `app/requirements.txt` (including `omnivoice`).
- Validated against **`omnivoice>=0.1.3,<0.2`** and **`gradio>=5.0,<7`**. These pins live in `app/requirements.txt` — bump them there and re-run **Install** or **Update** when upgrading.
- Starts the **Gradio** UI from **`app/app.py`** (same flow as the [Hugging Face Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)).
- Uses **one** Pinokio virtual environment: **`env/`** at the **project root** (not under `app/`). If you see both `env/` and `app/env/`, remove `app/env/` and rely on **Install** so PyTorch and packages live in the root `env/` only.

## Using in Pinokio

1. **Install** — creates `env/`, runs `uv pip install -r app/requirements.txt`, then `torch.js`.
2. **Start** — from the project root runs **`python app/app.py`** with **`OMNIVOICE_PORT={{port}}`** (next free port). Gradio binds to **`127.0.0.1`** by default; see environment variables below.
3. **Update** — `git pull` (if this folder is a git repo), then `uv pip install -U -r app/requirements.txt`.
4. **Reset** — removes `env/` for a clean reinstall.
5. **Save Disk Space** — `link.js` deduplicates venv library files.

## Environment variables (`app/app.py`)

| Variable | Purpose |
|----------|---------|
| `OMNIVOICE_MODEL` | Hugging Face repo or checkpoint (default `k2-fsa/OmniVoice`). |
| `OMNIVOICE_DEVICE` | Force `cuda`, `mps`, or `cpu` (default: auto cuda → mps → cpu). |
| `OMNIVOICE_LOAD_ASR` | `0` / `false` to skip Whisper ASR (less VRAM; supply reference text for clone). |
| `OMNIVOICE_HOST` | Gradio `server_name` (default **`127.0.0.1`**). Use `0.0.0.0` to listen on all interfaces. |
| `OMNIVOICE_PORT`, `PORT`, `GRADIO_SERVER_PORT` | Gradio port (Pinokio sets `OMNIVOICE_PORT`). |

### Hugging Face downloads (first run)

The first **Start** downloads checkpoints from the Hub (several GB). You may see:

- **`Warning: You are sending unauthenticated requests`** — set a [Hugging Face token](https://huggingface.co/settings/tokens) as **`HF_TOKEN`** in Pinokio (environment / secrets) or in your user environment so Hub rate limits are higher and downloads are faster.
- **Slow transfer** — this launcher sets **`HF_HUB_ENABLE_HF_TRANSFER=1`** in `start.js` (uses the `hf_transfer` package from `app/requirements.txt`).
- **Mirror (e.g. region / connectivity)** — optional: `HF_ENDPOINT` (see [OmniVoice README](https://github.com/k2-fsa/OmniVoice) for `hf-mirror` and similar).

### Device (CPU vs GPU)

Logs like **`Loading model ... to cpu`** mean PyTorch selected **CPU** (no usable CUDA/MPS, or CPU-only PyTorch). For **NVIDIA** acceleration, install with **Install** so `torch.js` installs the **CUDA** build, and ensure a recent GPU driver. You can force **`OMNIVOICE_DEVICE=cuda`** only if CUDA is actually available (`torch.cuda.is_available()`).

## Programmatic access

After **Start**, open the URL from Pinokio (“Open Web UI”) or the logs. The server is **`http://127.0.0.1:<port>`** by default.

### cURL

```bash
curl -sS "http://127.0.0.1:PORT/"
```

Replace `PORT` with the port shown in Pinokio.

### Python (Gradio client)

Use the [Gradio Python client](https://www.gradio.app/guides/getting-started-with-the-python-client) against `http://127.0.0.1:PORT` (same port as in the launcher).

### JavaScript

Use the [Gradio JavaScript client](https://www.gradio.app/guides/getting-started-with-the-python-client) (`@gradio/client`) with the same base URL.

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
