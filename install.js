module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        build: true,
        venv: "env",
        path: ".",
        message: ["uv pip install wheel"],
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: "uv pip install -r app/requirements.txt",
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
