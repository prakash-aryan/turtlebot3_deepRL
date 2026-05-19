#!/bin/bash
set -e

Xvnc :1 -rfbport 5901 -geometry 1600x900 -depth 24 -SecurityTypes None -localhost &
sleep 1
DISPLAY=:1 fluxbox &
websockify --web=/usr/share/novnc/ 6040 localhost:5901 &

exec "$@"
