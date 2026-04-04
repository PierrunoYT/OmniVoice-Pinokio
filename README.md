# OmniVoice (Pinokio)

[OmniVoice](https://github.com/k2-fsa/OmniVoice) zero-shot multilingual TTS (600+ languages) with voice cloning and voice design. Launcher scripts live in the repo root; application code is under `app/`.

## What it does

- Installs **PyTorch** via `torch.js` (per platform), then Python deps from `app/requirements.txt` (including `omnivoice`).
- Starts the **Gradio** UI from **`app/app.py`** (same flow as the [Hugging Face Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)).
- Uses a Pinokio-managed virtual environment **`env/`** at the project root.

## Using in Pinokio

1. **Install** — creates `env/`, runs `uv pip install -r app/requirements.txt`, then `torch.js`.
2. **Start** — runs `python app.py` in `app/` with **`OMNIVOICE_PORT={{port}}`** (next free port). Gradio binds to **`127.0.0.1`** by default; see environment variables below.
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

Optional: `HF_ENDPOINT` (e.g. mirror) if model download from Hugging Face is slow.

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
