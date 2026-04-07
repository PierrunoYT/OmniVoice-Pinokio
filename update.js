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
        message: "uv pip install -U -r app/requirements.txt",
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
