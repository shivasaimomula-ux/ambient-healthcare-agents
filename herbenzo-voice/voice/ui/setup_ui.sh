#!/usr/bin/env bash
# Fetch NVIDIA's WebRTC voice UI at the pinned ace-controller commit, apply Herbenzo config and branding,
# and install its dependencies into voice/ui/.build/webrtc_ui (git-ignored).
#
#   ./setup_ui.sh                 # prepare
#   cd .build/webrtc_ui && npm run dev -- --host 127.0.0.1 --port 4400
set -euo pipefail

ACE_COMMIT="cf371f0670b45ac387229259e43d6d56d78546e5"
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="$HERE/.build"
REPO="$BUILD/ace-controller"
UI="$BUILD/webrtc_ui"

mkdir -p "$BUILD"
if [ ! -d "$REPO/.git" ]; then
  git init -q "$REPO"
  git -C "$REPO" remote add origin https://github.com/NVIDIA/ace-controller.git
fi
if [ "$(git -C "$REPO" rev-parse HEAD 2>/dev/null || true)" != "$ACE_COMMIT" ]; then
  git -C "$REPO" fetch -q --depth 1 origin "$ACE_COMMIT"
  git -C "$REPO" checkout -q "$ACE_COMMIT"
fi

rm -rf "$UI"
cp -R "$REPO/examples/webrtc_ui" "$UI"
cp "$HERE/config.ts" "$UI/src/config.ts"

# Herbenzo branding: this is not an NVIDIA product page.
python3 - "$UI" <<'PY'
import sys
from pathlib import Path

ui = Path(sys.argv[1])
replacements = {
    ui / "src/App.tsx": [
        ('<img src="logo.png" alt="NVIDIA ACE Logo" className="h-16 mr-8" />', ""),
        ("Voice Agent Demo", "Herbenzo Health Intake"),
    ],
    ui / "index.html": [("<title>Speech to Speech Demo</title>", "<title>Herbenzo Health Intake</title>")],
    ui / "src/AudioWaveForm.tsx": [('lineColor = "#76B900", // NVIDIA green color', 'lineColor = "#2F7D4F",')],
}
for path, pairs in replacements.items():
    text = path.read_text()
    for old, new in pairs:
        if old not in text:
            sys.exit(f"branding patch failed: {old!r} not found in {path} (upstream changed?)")
        text = text.replace(old, new)
    path.write_text(text)
(ui / "public/logo.png").unlink(missing_ok=True)
PY

cd "$UI"
npm ci --no-audit --no-fund
echo "UI ready: cd $UI && npm run dev -- --host 127.0.0.1 --port 4400"
