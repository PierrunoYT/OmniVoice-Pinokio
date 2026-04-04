module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        env: {
          PYTHONUNBUFFERED: "1",
          OMNIVOICE_PORT: "{{port}}",
          HF_HUB_ENABLE_HF_TRANSFER: "1",
        },
        message: ["python app/app.py"],
        on: [
          {
            event: "/(http:\\/\\/[0-9.:]+)/",
            done: true,
          },
        ],
      },
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}",
      },
    },
  ],
}
