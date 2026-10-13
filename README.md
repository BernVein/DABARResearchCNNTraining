# sbc-feedtech — Fresh Raspberry Pi 5 Setup Guide

This is the full setup flow for taking a brand-new Raspberry Pi 5 (Pi OS Lite, arm64) from first boot to running the `sbc-feedtech` kiosk app. Each step includes _why_ it's there, not just the command.

**Default login:**

```
login: sbc-feedtech
pass:  DABAR01012002
```

---

## 1. Update the system

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

- `apt update` refreshes the local package index (list of what versions are available) — without this, apt often thinks packages/deps don't exist even though they do.
- `full-upgrade` (not just `upgrade`) also handles package removals/replacements needed by dependency changes, which is safer on a fresh image.
- Reboot after, since kernel/firmware updates usually need a fresh boot to take effect.

---

## 2. Set the system clock

```bash
sudo date -s "2026-08-17 12:00:00"
```

- Pi OS Lite has no RTC (real-time clock) battery by default, so on a fresh boot with no internet/NTP sync yet, the clock can be wildly wrong. This matters because SSH key generation, TLS/HTTPS requests (git clone, pip installs), and timestamps in logs can misbehave with an incorrect date.
- Once Wi-Fi is connected (next step), NTP will normally take over and keep time in sync automatically — this manual step is just to unblock things before that happens.

---

## 3. Connect to Wi-Fi

```bash
nmcli device wifi list
sudo nmcli device wifi connect "YOUR_SSID" password "YOUR_PASSWORD"
sudo reboot
```

- `nmcli` is NetworkManager's command-line tool — standard on Pi OS for headless Wi-Fi setup (no desktop GUI needed).
- `device wifi list` shows nearby networks so you can confirm the exact SSID name/spelling before connecting.
- Reboot afterward to make sure the connection persists cleanly and any network-dependent services start properly.

---

## 4. Fix tiny terminal text (if console font is too small)

This usually happens after certain `config.txt` edits (e.g. enabling the camera) change the console resolution, making the existing font look tiny.

```bash
sudo apt install -y xterm
resize
```

Or manually force terminal dimensions:

```bash
stty rows 50 cols 150
```

Then reconnect SSH to be safe:

```bash
exit
ssh <username>@<hostname>.local
```

- For a true font-size fix (not just resizing the terminal reporting), use:
  ```bash
  sudo dpkg-reconfigure console-setup
  ```
  and pick a larger font size (e.g. `16x32`) when prompted.

---

## 5. Install Picamera2 (camera support)

```bash
sudo apt install -y git python3-picamera2 --no-install-recommends
```

- `python3-picamera2` is the official camera stack for Pi OS — required since the backend (`main.py`) uses `Picamera2` directly for capture and streaming.
- `--no-install-recommends` keeps this lean by skipping optional extras (docs, GUI tools) not needed on a headless kiosk.
- `git` is installed here too since it's needed for the cloning steps next.

### Enable the camera overlay

The Pi doesn't auto-detect which camera sensor is attached — it needs to be told explicitly via a device tree overlay in the firmware config.

```bash
sudo nano /boot/firmware/config.txt
```

- Find the existing line `dtoverlay=vc4-kms-v3d` (this is the GPU driver overlay — leave it as-is).
- Directly below it, add a new line for your specific camera sensor:

```
  dtoverlay=imx219
```

Replace `imx219` with whatever sensor your camera module actually uses (e.g. `imx708` for Camera Module 3, `imx477` for HQ Camera). Check the label on your camera board or its documentation if you're not sure which one you have.

- Save and exit: `Ctrl+X` → `Y` → `Enter`.
- Reboot for the overlay to take effect:

```bash
  sudo reboot
```

---

## 6. Set up SSH keys for two GitHub accounts (backend + frontend)

Since the backend repo (`VSUResearch`) and frontend repo (`BernVein`) live under **different GitHub accounts**, you need **one SSH key per account** — SSH keys are tied to accounts, not individual repos.

### Generate the keys

```bash
# Backend key (VSUResearch account)
ssh-keygen -t ed25519 -C "sbc-feedtech-backend" -f ~/.ssh/id_ed25519_backend -N ""

# Frontend key (BernVein account)
ssh-keygen -t ed25519 -C "sbc-feedtech-frontend" -f ~/.ssh/id_ed25519_frontend -N ""
```

`-N ""` sets an empty passphrase — convenient for an unattended kiosk device that needs to `git pull` without a human typing a passphrase each time.

### Get the public keys to register on GitHub

```bash
cat ~/.ssh/id_ed25519_backend.pub            # add to VSUResearch account
cat ~/.ssh/id_ed25519_frontend.pub   # add to BernVein account
```

Add each one under the matching account's **Settings → SSH and GPG keys → New SSH key** (or as a repo-scoped **Deploy key** if you only want access to one specific repo, like was done for `sbc-feedtech` frontend).

### Configure SSH so each host alias uses the right key

```bash
nano ~/.ssh/config
```

```
# Account 1 (backend — VSUResearch)
Host github-backend
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_backend

# Account 2 (frontend — BernVein)
Host github-frontend
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_frontend
```

**How this works:** the `Host` value is just an alias _you_ invent — it doesn't have to be a real domain. When you `git clone` using that alias in the URL, SSH looks up the matching block and uses whichever `IdentityFile` (key) is listed there. This lets multiple accounts coexist on one device without conflicts.

### Reference

```
BACKEND:  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIH4EHXK6lQUonmPK/2DfVJur/EoornaK9i3HMOougfWs sbc-feedtech-backend
FRONTEND: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIG5qnZ+s6lMdPkR56kRUnS2GVtXs01K6vU26d0jl6tj8 sbc-feedtech-frontend
```

### Clone each repo using its matching alias

```bash
# Backend
git clone git@github-backend:VSUResearch/DABARResearchCNN.git DABARResearchCNN

# Frontend
git clone git@github-frontend:BernVein/sbc-feedtech.git sbc-feedtech
```

Using the wrong alias (e.g. `github.com` for a BernVein-only repo) will fail with a misleading **"repository not found"** error — GitHub deliberately doesn't distinguish "wrong key" from "repo doesn't exist" for private repos, as a security measure.

---

## 7. Create and activate the Python virtual environment

```bash
python3 -m venv ~/feedtech --system-site-packages
source ~/feedtech/bin/activate
```

- `--system-site-packages` is **important** — it lets the venv see globally-installed system packages (like `python3-picamera2`, which only installs cleanly via `apt`, not `pip`, since it needs to match the system's camera driver stack). Without this flag, the venv would be isolated from Picamera2 entirely and imports would fail.

---

## 8. Install Python dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- The Pi 5 has **no CUDA-capable GPU**, so installing plain `pip install torch torchvision` would pull the default CUDA-enabled build — multiple gigabytes larger than needed, and it wouldn't even use the GPU acceleration anyway since there's no NVIDIA hardware here. The `--index-url` explicitly points to the **CPU-only** build, which is smaller and correct for this hardware.

**If installs fail partway through** (common on slower/flaky connections):

```bash
pip install --upgrade pip
```

or add retry resilience for large packages like torch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --resume-retries 20
```

Then install the rest of the project's dependencies as needed (e.g. `fastapi`, `uvicorn`, `opencv-python-headless`, `ultralytics`, etc. — per `requirements.txt` or the reference environment).

---

## 9. Serve the frontend from the backend (no more separate frontend build step)

`main.py` mounts the built frontend directly as static files, on the same FastAPI app that serves the `/infer`, `/overlay`, `/health_check`, etc. endpoints:

```python
app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")
```

- This means one process, one port (`6767`), serves both the UI and the API — no separate static file server, no CORS juggling between two ports.
- `DIST_DIR` is expected to point at the frontend's built `dist/` folder as a sibling directory next to the backend repo. Confirm this path matches your actual folder layout on the Pi before running.

We are **no longer using Tauri**. The original plan was a Tauri `.deb` + WebKitGTK + `cage` (Wayland kiosk compositor) stack (see the old Step 9/10 below, kept for reference). That stack hit persistent rendering lag on this hardware that didn't reproduce when testing the same frontend in Chromium — so the deployment approach changed to: **build the frontend once (`npm run build`), serve it from the Python backend, and display it via Chromium in kiosk mode over a minimal X11 session**, dropping Tauri/WebKitGTK/cage entirely.

---

## 10. Install the minimal X11 + Chromium kiosk stack

```bash
sudo apt update
sudo apt install -y xserver-xorg x11-xserver-utils xinit chromium unclutter-xfixes
```

- `xserver-xorg` + `xinit` — the bare X server and the tool to start an X session from the terminal. No desktop environment or window manager is needed for a single-app kiosk.
- `x11-xserver-utils` — provides `xset` (screen blanking/DPMS control, so the display doesn't sleep on a kiosk).
- `chromium` — **not** `chromium-browser`; on this Debian Trixie-based Pi OS image the package (and binary) is just named `chromium`, at `/usr/bin/chromium`.
- `unclutter-xfixes` — was tried for cursor hiding, but ultimately **not what fixed it** (see Step 12 — X's own `-nocursor` flag was the actual fix, since touch input kept re-triggering unclutter's idle-based hiding).

---

## 11. Pin the display to the correct GPU device

The Pi 5 exposes **two** DRM devices under `/dev/dri` — `card0` and `card1` — plus a legacy `/dev/fb0` framebuffer. X can't reliably auto-pick the right one and fails with a framebuffer/busID error if left to guess.

Check which is which:

```bash
cat /sys/class/drm/card0/device/uevent
cat /sys/class/drm/card1/device/uevent
```

Look for the `DRIVER=` line: `v3d` is the 3D/render engine (not a display device); `vc4-drm` is the actual display/KMS driver. On this hardware, `card1` (`vc4-drm`) is the correct one to use for display output — confirm against your own output, as device numbering isn't guaranteed to be identical across units.

Pin it explicitly with a minimal Xorg config:

```bash
sudo mkdir -p /etc/X11/xorg.conf.d
sudo tee /etc/X11/xorg.conf.d/10-vc4.conf > /dev/null << 'EOF'
Section "Device"
    Identifier "vc4"
    Driver "modesetting"
    Option "kmsdev" "/dev/dri/card1"
EndSection
EOF
```

- `Driver "modesetting"` — the generic KMS driver recommended for the Pi's VC4/VC6 display stack (not the legacy `fbdev`/`fbturbo` drivers).
- `Option "kmsdev"` — explicitly tells X which device to use, removing the ambiguity that caused the earlier crash.
- This driver also enables `glamor` (GPU-accelerated 2D/X compositing) by default — confirmed working via `grep -i glamor /var/log/Xorg.0.log` showing `glamor X acceleration enabled on V3D`.

---

## 12. Write the kiosk launch script

Rather than passing a long Chromium flag list directly in the systemd unit, keep it in its own script for readability and easy tweaking:

```bash
sudo tee /home/sbc-feedtech/start-kiosk.sh > /dev/null << 'EOF'
#!/bin/bash
xset s off
xset s noblank
xset -dpms
setterm -blank 0 -powersave off -powerdown 0
exec /usr/bin/chromium \
  --kiosk \
  --noerrdialogs \
  --disable-infobars \
  --no-first-run \
  --use-gl=egl \
  --enable-gpu-rasterization \
  --ignore-gpu-blocklist \
  --enable-zero-copy \
  --disable-features=OverscrollHistoryNavigation,TouchpadOverscrollHistoryNavigation,TranslateUI \
  --disable-pinch \
  --disable-pull-to-refresh-effect \
  --disable-session-crashed-bubble \
  --autoplay-policy=no-user-gesture-required \
  http://localhost:6767
EOF
sudo chmod +x /home/sbc-feedtech/start-kiosk.sh
```

Why each flag is there:

- `--kiosk --noerrdialogs --disable-infobars --no-first-run` — baseline kiosk chrome removal: fullscreen, no error popups, no "restore pages"/update infobars, skip first-run setup screens.
- `--use-gl=egl` — forces Chromium to render through EGL/GLES against the vc4/v3d driver. Without this, Chromium can silently fall back to software rendering under X11.
- `--enable-gpu-rasterization` — moves rasterization work onto the GPU instead of the CPU.
- `--ignore-gpu-blocklist` — Chromium sometimes blocklists ARM/Pi GPUs as "unsupported" even when they work fine; this overrides that.
- `--enable-zero-copy` — avoids an extra CPU-side texture copy when compositing GPU-rendered content.
- `--disable-features=OverscrollHistoryNavigation,TouchpadOverscrollHistoryNavigation` — this is the flag that actually disables the swipe-left/right back/forward page navigation gesture. (Note: `--overscroll-history-navigation=0` is **not** a real Chromium switch and silently does nothing — this is the correct flag.)
- `--disable-pinch` / `--disable-pull-to-refresh-effect` — disables pinch-to-zoom and the touch drag-down-to-reload gesture, both unwanted on a fixed kiosk UI.
- `--disable-session-crashed-bubble` — suppresses the "Chromium didn't shut down correctly" popup that would otherwise appear after any hard power-cut reboot.
- `--autoplay-policy=no-user-gesture-required` — lets any video/camera stream elements autoplay without requiring a user tap first.
- `TranslateUI` (in the `--disable-features` list) — suppresses the translate-this-page prompt.

**Important — this must run as your normal user, not root.** Chromium has a hardcoded check that disables GPU acceleration entirely when running as UID 0, regardless of `--no-sandbox` or any other flag — confirmed by `chrome://gpu` showing every feature disabled when tested under `sudo`. This is part of why the systemd service (next step) runs as `User=sbc-feedtech` on the real console (`tty1`), not via an SSH+`sudo` session.

---

## 13. Fix `/dev/tty0` permissions (if testing manually first)

If you try running `xinit` by hand over SSH, you'll hit:

```
(EE) parse_vt_settings: Cannot open /dev/tty0 (Permission denied)
```

Two different causes can produce this — check both:

**A. Group membership.** `/dev/tty0` is normally owned `root:tty`. Check:

```bash
groups
ls -l /dev/tty0
```

If your user isn't in the `tty` group:

```bash
sudo usermod -a -G tty sbc-feedtech
sudo reboot
```

(Group changes only apply on a fresh login/session — a reboot is the reliable way to guarantee it.)

**B. SSH sessions have no "seat."** Even with correct group membership, an SSH session is a `pts/N` pseudo-terminal with no `seat0` assigned by `logind` (`loginctl` shows a blank `SEAT` column for it) — so dynamic device permissions for `/dev/tty0` are never granted to it at all, regardless of group membership. Only a session logged in directly at the physical console (`tty1`/`tty2`, shown with `seat0` in `loginctl`) gets that access.

**This is why the production setup uses a systemd service with `PAMName=login` + `TTYPath=/dev/tty1`** (Step 14) instead of manual `xinit` over SSH — it runs the whole thing as a real local console login, sidestepping both issues permanently.

---

## 14. Write a health-check wait script (avoids the kiosk/server race condition)

A fixed `sleep 5` before launching the kiosk is a guess, not a guarantee — if the server takes longer to come up (model loading, camera init), Chromium still launches on schedule and hits a blank page or 404 before the server is actually ready.

Instead, poll the server's own `/health_check` endpoint in a loop and only proceed once it responds successfully:

```bash
sudo tee /home/sbc-feedtech/wait-for-server.sh > /dev/null << 'EOF'
#!/bin/bash
URL="http://localhost:6767/health_check"
TIMEOUT=60
ELAPSED=0

while ! curl -sf "$URL" > /dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    echo "Server did not become healthy within ${TIMEOUT}s — exiting."
    exit 1
  fi
  sleep 1
  ELAPSED=$((ELAPSED + 1))
done

echo "Server is healthy after ${ELAPSED}s — launching kiosk."
EOF
sudo chmod +x /home/sbc-feedtech/wait-for-server.sh
```

- `curl -sf` — `-s` silences progress output; `-f` makes curl fail (non-zero exit) on HTTP error codes like 404/500, so the loop only exits successfully on a real 2xx response.
- Polls once per second, capped at 60s so a genuinely broken server doesn't hang the boot forever — after the timeout it exits non-zero, which fails the `ExecStartPre` step and prevents the kiosk from launching against a dead server.
- This script is referenced by `feedtech-kiosk.service`'s `ExecStartPre` in Step 15, replacing the old blind `sleep 5`.
- Note this only guarantees the _endpoint_ responds — if `/health_check` in `main.py` returns 200 immediately regardless of whether the camera/model are actually ready, the race condition can still exist. Confirm `/health_check` reflects true readiness, not just that the process is alive.

To watch the server's own startup output live (independent of the kiosk timing), open a separate session and tail its logs:

```bash
journalctl -u feedtech-server.service -f
```

---

## 15. Register the systemd services

Two services: one to keep `main.py` running, one to launch the kiosk display after it's up.

```bash
sudo tee /etc/systemd/system/feedtech-server.service > /dev/null << 'EOF'
[Unit]
Description=FeedTech Server
After=network.target

[Service]
Type=simple
User=sbc-feedtech
WorkingDirectory=/home/sbc-feedtech/DABARResearchCNN
ExecStart=/home/sbc-feedtech/feedtech/bin/python3 main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
```

- **`ExecStart` must point at the venv's own Python** (`~/feedtech/bin/python3`), **not** `/usr/bin/python3`. systemd services don't source your shell's profile/activation scripts, so the bare system Python has no access to the venv's installed packages (`cv2`, `torch`, etc.) — this caused a `ModuleNotFoundError: No module named 'cv2'` crash loop until corrected.

```bash
sudo tee /etc/systemd/system/feedtech-kiosk.service > /dev/null << 'EOF'
[Unit]
Description=FeedTech Chromium Kiosk
After=feedtech-server.service
Requires=feedtech-server.service
Conflicts=getty@tty1.service

[Service]
Type=simple
User=sbc-feedtech
PAMName=login
TTYPath=/dev/tty1
TTYReset=yes
TTYVHangup=yes
StandardInput=tty
StandardOutput=journal
StandardError=journal
ExecStartPre=/home/sbc-feedtech/wait-for-server.sh
ExecStart=/usr/bin/xinit /home/sbc-feedtech/start-kiosk.sh -- :0 vt1 -nocursor
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
```

- `Requires=feedtech-server.service` — ties the kiosk's lifecycle to the server's. Note this also means **if the server crash-loops, the kiosk gets torn down with it every cycle** — this is exactly what the `cv2` import error above caused, and is worth remembering as a diagnostic clue in future ("kiosk keeps restarting" often actually means "server keeps restarting").
- `PAMName=login` + `TTYPath=/dev/tty1` + `Conflicts=getty@tty1.service` — this is what makes the service behave like a real local console login on `tty1` (getting proper seat/VT permissions automatically via PAM) instead of a plain background process, while also preventing the normal login prompt (`getty`) from fighting X for the same terminal.
- `ExecStartPre=/home/sbc-feedtech/wait-for-server.sh` — replaces the old blind `sleep 5`. This actively polls `/health_check` (Step 14) and only lets the kiosk launch once the server responds, removing the race condition where Chromium could load before the backend was actually ready.
- `ExecStart=... -nocursor` — this is the X server flag (passed after `--`, alongside the display `:0` and VT `vt1`) that hides the mouse cursor entirely at the X level. This ended up being the actual fix for the cursor — `unclutter-xfixes` (Step 10) was tried first but its idle-based hiding kept getting re-triggered by touch input treated as mouse movement; disabling the cursor at the X server itself sidesteps that entirely, which also makes sense given this is a touch-only kiosk with no legitimate use for a pointer.

---

## 16. Enable and start both services

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now feedtech-server.service
sudo systemctl enable --now feedtech-kiosk.service
```

Confirm both are healthy (not stuck in `activating (auto-restart)`):

```bash
sudo systemctl status feedtech-server.service
sudo systemctl status feedtech-kiosk.service
```

If something's wrong, check the actual error/traceback:

```bash
journalctl -u feedtech-server.service -n 30 --no-pager
journalctl -u feedtech-kiosk.service -n 50 --no-pager
```

To confirm the wait script actually worked (server was healthy before kiosk launched), check the kiosk service's own log for the "Server is healthy after Ns" line:

```bash
journalctl -u feedtech-kiosk.service -n 30 --no-pager
```

Once both are `enabled` and healthy, a full `sudo reboot` should bring the Pi up straight into the fullscreen kiosk app with no manual steps — this is the actual test of whether the setup is solid, not just that the services started once.

---

## 17. Grant passwordless permission for power control (shutdown / reboot / sleep)

The kiosk UI has an on-screen power menu (Sleep / Reboot / Shutdown) backed by three FastAPI endpoints in `main.py`. Since `feedtech-server.service` runs as the unprivileged user `sbc-feedtech` (not root — see Step 12's note on why Chromium/the whole stack avoids root), that user needs a narrow, explicit exception to run just the power commands as root, without a password. `sudo shutdown`/`sudo systemctl suspend` would otherwise hang forever waiting for a password prompt that has nowhere to go, since a systemd service has no interactive TTY.

Create a dedicated sudoers file scoped to exactly two binaries — not blanket sudo access:

```bash
echo "sbc-feedtech ALL=(root) NOPASSWD: /sbin/shutdown, /usr/bin/systemctl suspend, /usr/bin/systemctl restart feedtech-server.service, /usr/bin/systemctl restart feedtech-kiosk.service" | sudo tee /etc/sudoers.d/feedtech-power
sudo chmod 0440 /etc/sudoers.d/feedtech-power
```

- Scoped to `/sbin/shutdown` and `/usr/bin/systemctl suspend` only — even if the API were somehow compromised, the exposure is "can power-cycle the Pi," not "can run arbitrary commands as root."
- `chmod 0440` matches the permissions sudoers expects for files under `/etc/sudoers.d/` (root-owned, read-only) — a misconfigured mode here can make `sudo` itself refuse to parse the file.
- This applies immediately — no service restart or reboot needed for the sudoers rule itself.

**Verify it took effect**, without actually shutting anything down:

```bash
sudo -n /sbin/shutdown -c        # cancels any pending shutdown; harmless no-op if none scheduled
sudo -n -l | grep suspend        # should list /usr/bin/systemctl suspend under NOPASSWD
```

If both return instantly with no password prompt, the rule is working. If either hangs or asks for a password, double check the username in the sudoers file matches the actual service user (`grep User= /etc/systemd/system/feedtech-server.service`).

---

## 18. Testing & troubleshooting reference

Quick checks for the pieces that tend to need re-verifying after a fresh setup, a reboot, or after pulling new code.

### Camera

Check the camera is detected and actually working at the OS level (independent of Picamera2/Python):

```bash
libcamera-hello --list-cameras
```

- Should list your sensor (e.g. `imx219`) with its supported resolutions/formats. If it prints nothing or errors, the issue is at the `dtoverlay` / hardware level (Step 5), not in `main.py` — no point debugging Python code until this works.
- Quick live-preview sanity check (if you have a display attached, skip on a headless kiosk):
  ```bash
  libcamera-hello -t 5000
  ```
  Shows a 5-second preview window. On headless setups, use the still-capture check instead:
  ```bash
  libcamera-still -o /tmp/test.jpg
  ```
  then pull `/tmp/test.jpg` off the Pi (`scp`) to confirm it actually captured something.

Confirm the Python package itself is installed and importable:

```bash
python3 -c "from picamera2 import Picamera2; print('ok')"
```

Run this both with plain `python3` and with the venv's Python (`~/feedtech/bin/python3`) — a mismatch between the two (works in one, not the other) points back at the `--system-site-packages` venv flag from Step 7.

### Services — restart / status / logs

Restart either service (e.g. after pulling new backend code, or after any systemd unit file edit):

```bash
sudo systemctl restart feedtech-server.service
sudo systemctl restart feedtech-kiosk.service
```

After editing a `.service` file itself, reload systemd's config before restarting:

```bash
sudo systemctl daemon-reload
sudo systemctl restart feedtech-server.service
```

Check current status at a glance:

```bash
sudo systemctl status feedtech-server.service
sudo systemctl status feedtech-kiosk.service
```

Look for `active (running)` — anything showing `activating (auto-restart)` repeatedly means it's crash-looping (see Step 15's note: a crash-looping server also tears down the kiosk, since the kiosk `Requires=` it).

Tail live logs while reproducing an issue:

```bash
journalctl -u feedtech-server.service -f
journalctl -u feedtech-kiosk.service -f
```

Pull the last N lines without following (good for "what just happened"):

```bash
journalctl -u feedtech-server.service -n 50 --no-pager
```

Stop a service temporarily without disabling it on boot (useful when you want to run `main.py` manually in the foreground to see errors directly):

```bash
sudo systemctl stop feedtech-server.service
cd /home/sbc-feedtech/DABARResearchCNN
/home/sbc-feedtech/feedtech/bin/python3 main.py
```

Remember to `sudo systemctl start feedtech-server.service` again afterward so it goes back to being managed normally.

### Backend endpoint sanity checks

Confirm the server is up and responding, without going through the kiosk UI at all:

```bash
curl -sf http://127.0.0.1:6767/health_check
```

Check the power endpoints directly (same caution as always — these will actually reboot/shutdown/sleep the Pi):

```bash
curl -X POST http://127.0.0.1:6767/system/reboot
curl -X POST http://127.0.0.1:6767/system/sleep
curl -X POST http://127.0.0.1:6767/system/shutdown
```

If these hang instead of returning immediately, it's almost always the sudoers rule (Step 17) — re-run the verification commands there before looking anywhere else.

### Display / kiosk

If the kiosk service is running but the screen is blank or shows an X error, check the Xorg log for the actual failure:

```bash
cat /var/log/Xorg.0.log | grep -i "EE"
```

Confirm the GPU driver pinning (Step 11) took effect:

```bash
grep -i glamor /var/log/Xorg.0.log
```

Should show `glamor X acceleration enabled on V3D` — if it's missing or shows a fallback, re-check `/etc/X11/xorg.conf.d/10-vc4.conf` and which `card` is actually the display device on this unit (`card0`/`card1` numbering isn't guaranteed identical across units).

---

## 19. Potential issues after install

### Stale Chromium cache serving old dev-mode assets (`/src/main.css` 404)

If you switch the frontend from `npm run dev` to a production build (`npm run build`) and the kiosk still shows a blank/broken page, check the backend logs:

```bash
journalctl -u feedtech-server.service -n 15 --no-pager
```

If you see a 404 for `/src/main.css` (a Vite **dev-server** path, not a production build path), the served `dist/index.html` itself may already be correct — Chromium's **HTTP disk cache** can be holding onto the old dev-mode page independently of the profile config.

Two separate locations are involved, and clearing only one isn't enough:

- `~/.config/chromium` — profile config, cookies, local storage, settings
- `~/.cache/chromium` — the actual HTTP disk cache (page/asset cache), stored separately per the XDG cache convention

Clear **both**, with the kiosk service stopped first:

```bash
sudo systemctl stop feedtech-kiosk.service
rm -rf ~/.config/chromium ~/.cache/chromium
sudo systemctl start feedtech-kiosk.service
```

Then re-check the logs — you should see requests for hashed `/assets/...` files instead of `/src/main.css`.

## 20. Install Plymouth

Pi OS Lite does **not** ship with Plymouth by default (it's normally bundled only with the desktop image), so it needs to be installed manually.

```bash
sudo apt update
sudo apt install -y plymouth plymouth-themes rpd-plym-splash
```

- `plymouth` — the core daemon/engine that actually renders the splash during boot.
- `plymouth-themes` — a bundle of ready-made themes, including `spinner` (plain circular spinner) and `spinfinity` (spinner in the shape of an infinity sign). This is what we'll use instead of hand-building an animation from scratch.
- `rpd-plym-splash` — **this one is easy to miss, but required.** `raspi-config`'s splash screen toggle specifically checks for the file `/usr/share/plymouth/themes/pix/pix.script`, which only comes from this package (not from `plymouth-themes`). Without it, `raspi-config` will refuse to enable the splash screen at all, with the error _"The splash screen is not installed so cannot be activated."_
  **How to check it installed correctly:**

```bash
ls /usr/share/plymouth/themes/spinner/
ls /usr/share/plymouth/themes/pix/pix.script
```

The first should list a bunch of `throbber-*.png` files plus `spinner.plymouth`. The second should print the file path back with no "No such file" error.

---

## 21. Enable the splash screen

```bash
sudo raspi-config
```

Navigate to: `System Options → Boot / Auto Login → Splash Screen → Yes`

Exit and reboot to confirm it's active:

```bash
sudo reboot
```

**How to check:** during boot you should see _some_ graphical splash (even the default Raspberry Pi logo/rainbow at this point) instead of plain scrolling text. If you still get the "not installed" error here, Step 20 didn't complete correctly — re-check that `rpd-plym-splash` is actually installed.

---

## 22. Switch to the `spinner` theme (plain circular loading animation)

By default, Pi OS uses the `pix` theme (static logo, dot-style progress indicator, no true spinning circle). Swap to `spinner` for a plain rotating circle, closer to a Windows-style loading indicator.

```bash
sudo plymouth-set-default-theme -R spinner
sudo reboot
```

- `plymouth-set-default-theme <name>` sets which theme is active.
- The `-R` flag additionally **rebuilds the initramfs** in the same step. This matters because the splash screen's theme, images, and config are all baked into a compressed initramfs image that loads very early in boot (before the real root filesystem is even mounted) — simply changing files in `/usr/share/plymouth/themes/` on disk does **nothing** on its own. Every single change from here on (logo swap, config edit, spinner resize) requires re-running this rebuild step, or the change won't show up on the next boot.
  **How to check current theme:**

```bash
plymouth-set-default-theme
```

(Running it with no arguments prints the currently active theme name.)

**How to check available themes:**

```bash
plymouth-set-default-theme -l
```

---

## 23. Add your logo (the "watermark")

The `spinner` theme supports an optional logo image, but doesn't ship with one by default — the file just doesn't exist yet in the theme folder. Plymouth expects it at an exact filename:

```
/usr/share/plymouth/themes/spinner/watermark.png
```

**Before copying it in — two things matter about the source image itself:**

1. **Transparency:** Plymouth renders PNG alpha transparency correctly, and the `spinner` theme's background is solid black, so a logo with a transparent background will blend in cleanly with no visible box/rectangle around it. Export your logo as a PNG with transparency preserved.
2. **Size:** the `two-step` plugin (which `spinner` uses) has **no scaling option** — it always draws the image at its exact native pixel dimensions. If your source logo is large (e.g. a wide banner-style export), it needs to be resized down _before_ copying it in, or it'll dominate/overflow the screen. See Step 24 for resizing.
   **Copy the logo in and rebuild:**

```bash
sudo cp ~/path/to/your-logo.png /usr/share/plymouth/themes/spinner/watermark.png
sudo plymouth-set-default-theme -R spinner
sudo reboot
```

To update the logo again later (new version, same or different filename), it's the exact same two commands — `cp` always just overwrites whatever is currently at `watermark.png`, and the `-R` rebuild is what actually pushes the new file into effect on the next boot.

**How to check the logo file is actually in place and its real pixel size:**

```bash
ls -la /usr/share/plymouth/themes/spinner/watermark.png
identify /usr/share/plymouth/themes/spinner/watermark.png
```

(`identify` comes from ImageMagick — install with `sudo apt install -y imagemagick` if it's not already present from Step 24.)

---

## 24. Resize the logo (if it's too big)

Since there's no config-based scaling, the fix is to physically shrink the image file.

```bash
sudo apt install -y imagemagick

convert ~/path/to/your-logo.png -resize 300x300 ~/path/to/your-logo-small.png

sudo cp ~/path/to/your-logo-small.png /usr/share/plymouth/themes/spinner/watermark.png
sudo plymouth-set-default-theme -R spinner
sudo reboot
```

- `-resize 300x300` is a **bounding box**, not a forced square crop — the image scales down proportionally to fit within 300×300px, keeping its original aspect ratio intact. Adjust the number based on your screen resolution and how big you want the logo to appear.

---

## 25. Reposition the logo

Position is controlled through the theme's config file, separately from the spinner's own position.

```bash
sudo nano /usr/share/plymouth/themes/spinner/spinner.plymouth
```

Look for these two lines (add them under the `[two-step]` section if they don't already exist):

```
WatermarkHorizontalAlignment=.5
WatermarkVerticalAlignment=.25
```

- Both values range from `0` to `1`.
- `WatermarkHorizontalAlignment=.5` = perfectly centered left-to-right (leave this alone).
- `WatermarkVerticalAlignment` controls vertical position: `0` = very top, `1` = very bottom, `.5` = vertical center. The theme's default is `.96` (near-bottom) — lower this number to move the logo up. Try `.25`–`.35` for "upper area, not glued to the edge."
  Save in nano: `Ctrl+O`, `Enter` to confirm, then `Ctrl+X` to exit. **Double-check you actually saved** — exiting nano without confirming the write (e.g. accidentally hitting "N" on a save prompt) is a common reason an edit silently doesn't take effect.

Rebuild and check:

```bash
sudo plymouth-set-default-theme -R spinner
sudo reboot
```

**How to check your edit was actually saved to disk** (before even rebooting):

```bash
grep Watermark /usr/share/plymouth/themes/spinner/spinner.plymouth
```

---

## 26. Resize the spinner animation (if it's too small)

Same limitation as the logo — no config option controls spinner size, only position (`HorizontalAlignment` / `VerticalAlignment` in the same file). The only way to make it visually bigger is to resize the actual animation frame images.

**Always back up the original theme folder first**, so you can revert if needed:

```bash
sudo cp -r /usr/share/plymouth/themes/spinner /usr/share/plymouth/themes/spinner-backup
```

Batch-resize all 30 spinner frames at once:

```bash
sudo mogrify -resize 200x200 /usr/share/plymouth/themes/spinner/throbber-*.png
```

- `mogrify` is ImageMagick's batch/in-place version of `convert` — it overwrites each matched file directly rather than producing new output files, which is what lets one command resize all 30 frames together (keeping them proportionally consistent, which matters for a smooth-looking animation).
- Check the current frame size first if you want a size reference point: `identify /usr/share/plymouth/themes/spinner/throbber-0001.png`
  Rebuild and check:

```bash
sudo plymouth-set-default-theme -R spinner
sudo reboot
```

**To revert back to the original size** if you don't like the result:

```bash
sudo rm -rf /usr/share/plymouth/themes/spinner
sudo cp -r /usr/share/plymouth/themes/spinner-backup /usr/share/plymouth/themes/spinner
sudo plymouth-set-default-theme -R spinner
sudo reboot
```

---

## 27. Keep the splash on screen with zero blank gap (wait for the real server, not just "boot")

**The problem:** Plymouth has no idea your kiosk app exists. It automatically quits once core boot reaches `multi-user.target` — completely independent of whether `feedtech-server.service` has actually finished loading its model, or whether Chromium has actually painted the page. That timing gap between "Plymouth quits" and "kiosk actually appears" is what shows up as a blank screen.

**The fix:** make Plymouth's own quit step wait on the same health check the kiosk service already uses (`wait-for-server.sh` from Step 14 of the main guide), instead of quitting on a fixed schedule.

### 27a. Wrap the health check in its own "ready" gate service

```bash
sudo tee /etc/systemd/system/feedtech-ready.service > /dev/null << 'EOF'
[Unit]
Description=Wait until FeedTech server is healthy
After=feedtech-server.service
Requires=feedtech-server.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/home/sbc-feedtech/wait-for-server.sh

[Install]
WantedBy=multi-user.target
EOF
```

- `Type=oneshot` — this service runs the script once and exits, rather than staying running as a background process.
- `RemainAfterExit=yes` — even though the process exits, systemd keeps marking the service as "active" afterward. This is what lets other services check "has this already succeeded?" rather than "is this still running?"
- `Requires=` / `After=feedtech-server.service` — this gate can't even start running its check until your actual server service has started.

### 27b. Reload systemd and enable it

```bash
sudo systemctl daemon-reload
sudo systemctl enable feedtech-ready.service
```

- `daemon-reload` — tells systemd to re-scan `/etc/systemd/system/` for new or changed `.service` files. **Required any time you create or edit a unit file** — systemd won't pick up the change otherwise, even on reboot, because it caches unit definitions in memory.
- `enable` — makes it start automatically on every future boot (not just when triggered manually). Without this, it would only run if you started it by hand with `systemctl start`.
  **How to check it manually before trusting it on a real boot:**

```bash
sudo systemctl start feedtech-ready.service
sudo systemctl status feedtech-ready.service
```

Look for `Active: active (exited)` and `status=0/SUCCESS` in the output — that's the expected healthy end-state for a `oneshot` + `RemainAfterExit=yes` service. If your server wasn't already running/healthy, this command will actually sit and wait (since it's running the real health-check loop) until the server responds or the script's 60-second timeout is hit.

### 27c. Make Plymouth's quit step depend on the ready gate

Rather than editing Plymouth's own built-in unit file directly (risky — package updates can silently overwrite it), use a systemd **drop-in override**, which layers additional config on top without touching the original file:

```bash
sudo mkdir -p /etc/systemd/system/plymouth-quit.service.d
sudo tee /etc/systemd/system/plymouth-quit.service.d/wait-for-feedtech.conf > /dev/null << 'EOF'
[Unit]
After=feedtech-ready.service
Requires=feedtech-ready.service
EOF
```

- `After=` — `plymouth-quit.service` must start _after_ `feedtech-ready.service` finishes.
- `Requires=` — `plymouth-quit.service` is not allowed to run at all unless `feedtech-ready.service` succeeded first.
- Together: Plymouth physically cannot be dismissed until your server-health check has passed.
  Reload systemd again so it picks up this new drop-in file:

```bash
sudo systemctl daemon-reload
```

**How to check the override is actually registered:**

```bash
systemctl cat plymouth-quit.service
```

This prints the _effective_, merged configuration — you should see your `wait-for-feedtech.conf` content listed underneath the original unit file's content, under a line noting the drop-in path.

### 27d. Full reboot test

```bash
sudo reboot
```

Watch the entire boot closely. You're checking for exactly one thing: the splash (logo + spinner) stays up **continuously**, with zero gap of blank/black/text screen, right up until the kiosk UI appears directly on top of it.

**If you still see a blank gap**, check what actually happened, in this order:

```bash
journalctl -u feedtech-ready.service -b --no-pager
journalctl -u plymouth-quit.service -b --no-pager
```

- `-b` limits output to the current boot only, which matters here since you specifically want to see what happened during the boot you just watched, not a stale log entry from an earlier one.
- In the `feedtech-ready` log, confirm you see the `"Server is healthy after Ns"` line and that it happened at a reasonable point in boot (not suspiciously early, which could mean it's checking a stale/cached health response instead of a genuine fresh startup).
- In the `plymouth-quit` log, confirm its start timestamp comes _after_ `feedtech-ready`'s finish timestamp — if it started before, the `After=`/`Requires=` override isn't actually being applied, and it's worth re-checking Step 27c's `systemctl cat` output.

---

## Quick reference — commands you'll reuse constantly

| Task                                            | Command                                                   |
| ----------------------------------------------- | --------------------------------------------------------- |
| See current theme                               | `plymouth-set-default-theme`                              |
| List available themes                           | `plymouth-set-default-theme -l`                           |
| Switch theme + rebuild                          | `sudo plymouth-set-default-theme -R <name>`               |
| Rebuild without switching                       | `sudo plymouth-set-default-theme --rebuild-initrd <name>` |
| Reload systemd after any `.service` file change | `sudo systemctl daemon-reload`                            |
| Check a service's live status                   | `sudo systemctl status <name>.service`                    |
| See a service's merged config (incl. drop-ins)  | `systemctl cat <name>.service`                            |
| Logs from the current boot only                 | `journalctl -u <name>.service -b --no-pager`              |
| Check current image dimensions                  | `identify <file>.png`                                     |
| Resize one image (proportional)                 | `convert in.png -resize WxH out.png`                      |
| Resize many images in place                     | `sudo mogrify -resize WxH /path/*.png`                    |

## Notes for a future "golden image" / cloning to other Pi units

- Steps 1–5 are OS-level setup, generic to any Pi OS Lite install.
- Step 6 (SSH keys) is **per-device** — each physical Pi unit should get its own key pair, registered individually on GitHub, rather than copying one key across multiple units (for auditability and easy revocation if a unit is lost/compromised).
- Steps 7–8 (venv + deps) and 9–16 (serve + kiosk) are project-specific and can be scripted/automated further (e.g. a single `setup.sh`) once the flow is stable.
- The `/etc/X11/xorg.conf.d/10-vc4.conf` device pinning (Step 11) should be double-checked on each new unit — `card0`/`card1` numbering is not guaranteed to be identical across units or firmware versions; always verify via `uevent` before assuming `card1` is correct.
- We are intentionally **not** using Tauri, WebKitGTK, or `cage` anymore — see Step 9 for why. Do not resurrect the old `.deb`/`install-pi.sh` flow unless deliberately reverting this decision.
- Step 14 (`wait-for-server.sh`) replaces the earlier fixed `sleep 5` approach — carry this script and the updated `ExecStartPre` line forward when cloning to new units, rather than reverting to a blind sleep.
- Step 17 (power-control sudoers rule) is **per-device**, same reasoning as Step 6 — each unit's service user needs its own `/etc/sudoers.d/feedtech-power` file; it isn't something that ships inside the git repos.
