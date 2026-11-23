#!/usr/bin/env bash
# Launch Stable Diffusion WebUI with API and CORS enabled for local frontend usage.
SD_DIR="${SD_WEBUI_DIR:-$HOME/sd/stable-diffusion-webui}"
cd "$SD_DIR" || exit 1
./venv/bin/python3 webui.py \
  --opt-sdp-attention \
  --api \
  --listen \
  --cors-allow-origins "http://127.0.0.1:8000,http://localhost:8000" \
  --skip-python-version-check
