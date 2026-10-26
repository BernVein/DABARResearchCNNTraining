# FeedTech Raspberry Pi Setup Guide

This guide organizes the supplied notepad into a deployment procedure. All commands and configuration snippets come from the original notes.

## Prerequisites

- Use the deployment account `sbc-feedtech`, with home directory `/home/sbc-feedtech`.
- Clone both repositories into that home directory.
- Configure the SSH aliases `github-backend` and `github-frontend` for their respective keys before cloning.
- Prepare the frontend build in `/home/sbc-feedtech/sbc-feedtech/dist` before configuring Nginx.
- Make required application settings and model files available to the backend.

## 1. Update the system and install packages

Update installed packages and reboot:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

Reconnect after the reboot, then install the required packages:

```bash
sudo apt install -y git python3-picamera2 hailo-all nginx --no-install-recommends
```

Verify installation:

```bash
for p in git python3-picamera2 hailo-all nginx; do dpkg -s $p >/dev/null 2>&1 && echo "$p: installed" || echo "$p: MISSING"; done
```

Each listed package should report `installed`.

## 2. Configure the IMX219 camera

Open the firmware configuration:

```bash
sudo nano /boot/firmware/config.txt
```

Add the camera overlay:

```ini
dtoverlay=imx219
```

Save the file. A newly added firmware overlay takes effect after a reboot; perform the check below after the next reboot if needed.

```bash
rpicam-hello --list-cameras
```

Confirm that the intended camera appears. Camera configuration depends on the hardware and connected port.

## 3. Generate and register deploy keys

Create separate keys for the backend and frontend repositories:

```bash
ssh-keygen -t ed25519 -C "sbc-feedtech-backend" -f ~/.ssh/id_ed25519_backend -N ""
ssh-keygen -t ed25519 -C "sbc-feedtech-frontend" -f ~/.ssh/id_ed25519_frontend -N ""
```

Display the public keys:

```bash
cat ~/.ssh/id_ed25519_backend.pub
cat ~/.ssh/id_ed25519_frontend.pub
```

Add each public key to the matching GitHub repository's deploy keys. Keep private keys on the device. Avoid overwriting existing keys when repeating setup.

## 4. Clone the repositories

### Configure SSH so each host alias uses the right key

```bash
nano ~/.ssh/config
```

```
# Account 1 (backend - VSUResearch)
Host github-backend
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_backend

# Account 2 (frontend - BernVein)
Host github-frontend
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_frontend
```

Run these commands from `/home/sbc-feedtech` so the resulting paths match the service configuration:

```bash
git clone git@github-backend:VSUResearch/DABARResearchCNN.git DABARResearchCNN
git clone git@github-frontend:BernVein/sbc-feedtech.git sbc-feedtech
```

## 5. Prepare the Python environment

Create and activate the required virtual environment:

```bash
python3 -m venv ~/feedtech --system-site-packages
source ~/feedtech/bin/activate
pip install --upgrade pip
```

The environment can access OS-installed Python packages through `--system-site-packages`.

Then install required packages using `requirements.txt`

```bash
python -m pip install -r requirements.txt
```

### Or do it manually if some error occurs

Install PyTorch and Torchvision using the CPU package index:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

The setup selects CPU packages to limit unnecessary dependency downloads and storage use for the YOLO workflow.

Install the remaining dependencies while the environment is active:

```bash
pip install uvicorn fastapi opencv-python-headless ultralytics onnxruntime rich pymongo python-multipart numpy
```

Verify installation:

```bash
source ~/feedtech/bin/activate

for p in uvicorn fastapi opencv-python-headless ultralytics onnxruntime rich pymongo python-multipart numpy torch torchvision; do
  pip show "$p" >/dev/null 2>&1 && echo "$p: installed" || echo "$p: MISSING"
done

python -c "import torch; print('Version:', torch.__version__); print('CUDA build:', torch.version.cuda)"
```

## 6. Install the kiosk display stack

```bash
sudo apt install -y xserver-xorg x11-xserver-utils xinit chromium unclutter-xfixes
```

Verify installation:

```bash
for p in xserver-xorg x11-xserver-utils xinit chromium unclutter-xfixes; do dpkg -s $p >/dev/null 2>&1 && echo "$p: installed" || echo "$p: MISSING"; done
```

| Package             | Purpose                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| `xserver-xorg`      | Provides the X11 display server.                                                      |
| `x11-xserver-utils` | Provides display utilities, including screen blanking and output configuration tools. |
| `xinit`             | Starts an X session without requiring a full desktop environment.                     |
| `chromium`          | Displays the frontend in kiosk mode.                                                  |
| `unclutter-xfixes`  | Provides cursor hiding when launched. The supplied launcher does not launch it.       |

## 7. Configure the display device if required

Use this configuration when Xorg cannot select the display device in the original setup. The path `/dev/dri/card1` is hardware-specific and must match the target device.

```bash
sudo mkdir -p /etc/X11/xorg.conf.d
sudo nano /etc/X11/xorg.conf.d/10-vc4.conf
```

Paste:

```xorg
Section "Device"
    Identifier "vc4"
    Driver "modesetting"
    Option "kmsdev" "/dev/dri/card1"
EndSection
```

Test from the Raspberry Pi's physical console, not an SSH session:

```bash
startx
```

A graphical session indicates that X can start. The session displayed depends on the installed software and configuration.

## 8. Configure Nginx

Ensure the frontend build exists in the expected `dist` directory. Open the default site configuration:

```bash
sudo nano /etc/nginx/sites-available/default
```

Replace the existing document root with:

```nginx
root /home/sbc-feedtech/sbc-feedtech/dist;
```

Allow directory traversal, validate the configuration, enable Nginx at boot, and reload it:

```bash
chmod o+x /home/sbc-feedtech /home/sbc-feedtech/sbc-feedtech
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl reload nginx
curl -I http://localhost
```

Confirm that validation succeeds and the local site responds. Enabling Nginx configures future startup; reloading assumes it is already running. The permission change grants directory traversal through the listed directories.

## 9. Create the kiosk launcher

Open the launcher file:

```bash
nano ~/start-kiosk.sh
```

Paste:

```bash
#!/bin/bash

# wait until nginx answers
until curl -sf http://localhost > /dev/null; do
  sleep 1
done

# screen stays awake
xset s off
xset s noblank
xset -dpms

# start chromium
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
  http://localhost
```

Make the launcher executable:

```bash
chmod +x ~/start-kiosk.sh
```

The launcher waits for Nginx, disables screen blanking, and opens Chromium in kiosk mode. Its wait loop has no timeout. A responding frontend does not establish backend readiness.

## 10. Start the kiosk automatically at boot

Create the kiosk service:

```bash
sudo nano /etc/systemd/system/feedtech-kiosk.service
```

Paste:

```ini
[Unit]
Description=FeedTech Chromium Kiosk
After=network.target nginx.service
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
ExecStart=/usr/bin/xinit /home/sbc-feedtech/start-kiosk.sh -- :0 vt1 -nocursor
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

This service uses `tty1` and display `:0`. Ensure another graphical session is not already using those resources.

Register the service, enable startup, and reboot:

```bash
sudo systemctl daemon-reload
sudo systemctl enable feedtech-kiosk.service
sudo reboot
```

Confirm that the frontend opens in kiosk mode after startup.

## 11. Start the backend automatically at boot

Create the backend service:

```bash
sudo nano /etc/systemd/system/feedtech-server.service
```

Paste:

```ini
[Unit]
Description=FeedTech Backend (main.py)
After=network.target

[Service]
Type=simple
User=sbc-feedtech
WorkingDirectory=/home/sbc-feedtech/DABARResearchCNN
ExecStart=/home/sbc-feedtech/feedtech/bin/python main.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

The service runs `main.py` with the virtual environment's Python interpreter. Required settings must be available to the application; variables exported only in an interactive shell are not automatically inherited by a system service.

Register the service, enable startup, and reboot:

```bash
sudo systemctl daemon-reload
sudo systemctl enable feedtech-server.service
sudo reboot
```

Confirm that the frontend can communicate with the backend after startup.

## 12. Configure passwordless operational permissions

Apply the sudoers rule from the original notes:

```bash
echo "sbc-feedtech ALL=(root) NOPASSWD: /sbin/shutdown, /usr/bin/systemctl suspend, /usr/bin/systemctl restart feedtech-server.service, /usr/bin/systemctl restart feedtech-kiosk.service" | sudo tee /etc/sudoers.d/feedtech-power
sudo chmod 0440 /etc/sudoers.d/feedtech-power
```

This rule allows shutdown, suspend, and the specified service restarts without a password. The shutdown entry permits arguments to that executable; a separate direct reboot command is not listed. Application calls must use the permitted paths, and suspend requires device support.

## Completion checklist

- [ ] Required system packages are installed.
- [ ] The camera is detected after its configuration takes effect.
- [ ] Repositories are cloned into the expected directories.
- [ ] The Python environment and dependencies are installed.
- [ ] The frontend build exists at the Nginx document root.
- [ ] Nginx validation succeeds and the local site responds.
- [ ] The kiosk opens automatically after a reboot.
- [ ] The backend starts automatically after a reboot.
- [ ] Frontend requests and the camera/inference workflow work together.
- [ ] Operational permissions match the application's requirements.
