module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        message: "git pull",
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        // Upgrade only the app's direct deps (within app/requirements.txt pins).
        // A blanket -U would also replace the torch.js-installed PyTorch with
        // the default PyPI build (CPU-only on Windows) before torch.js reverts it.
        message: "uv pip install -r app/requirements.txt -P omnivoice -P transformers -P accelerate -P gradio -P pydub -P soundfile",
      },
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: ".",
        },
      },
    },
  ],
}
