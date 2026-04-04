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
        path: "app",
        message: "uv pip install -r requirements.txt",
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
