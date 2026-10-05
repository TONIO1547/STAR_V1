#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Lancer avec sudo: sudo bash scripts/install_jetson.sh"
  exit 1
fi

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="/mnt/sdcard/yolov5"
SERVICE="drone.service"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP_DIR="${TARGET_DIR}/backups/star-pid-${STAMP}"
ROLLBACK_READY=0

restore_backup() {
  install -o robot3 -g robot3 -m 0644 \
    "${BACKUP_DIR}/stream_drone_pid.py" \
    "${TARGET_DIR}/stream_drone_pid.py"
  if [[ -f "${BACKUP_DIR}/star_pid.py" ]]; then
    install -o robot3 -g robot3 -m 0644 \
      "${BACKUP_DIR}/star_pid.py" "${TARGET_DIR}/star_pid.py"
  fi
  for module in star_predictive_pid.py star_realtime.py; do
    if [[ -f "${BACKUP_DIR}/${module}" ]]; then
      install -o robot3 -g robot3 -m 0644 \
        "${BACKUP_DIR}/${module}" "${TARGET_DIR}/${module}"
    fi
  done
  if [[ -f "${BACKUP_DIR}/device_auth.py" ]]; then
    install -o robot3 -g robot3 -m 0644 \
      "${BACKUP_DIR}/device_auth.py" "${TARGET_DIR}/device_auth.py"
  fi
  if [[ -f "${BACKUP_DIR}/star-tracking.env" ]]; then
    install -o root -g root -m 0600 \
      "${BACKUP_DIR}/star-tracking.env" /etc/star-tracking.env
  fi
  systemctl restart "${SERVICE}" || true
}

on_error() {
  local status=$?
  if [[ "${ROLLBACK_READY}" -eq 1 ]]; then
    echo "Echec de l'installation, restauration automatique..."
    restore_backup
  fi
  exit "${status}"
}

trap on_error ERR

for required in \
  "${TARGET_DIR}/stream_drone_pid.py" \
  "${KIT_DIR}/stream_drone_pid.py" \
  "${KIT_DIR}/star_predictive_pid.py" \
  "${KIT_DIR}/star_realtime.py" \
  "${KIT_DIR}/device_auth.py" \
  "/etc/star-device.env"; do
  if [[ ! -f "${required}" ]]; then
    echo "Fichier requis absent: ${required}"
    exit 1
  fi
done

if ! systemctl cat "${SERVICE}" >/dev/null 2>&1; then
  echo "Le service ${SERVICE} n'existe pas. Utiliser le modele jetson/systemd/drone.service."
  exit 1
fi

install -d -o robot3 -g robot3 -m 0750 "${BACKUP_DIR}"
install -o robot3 -g robot3 -m 0644 \
  "${TARGET_DIR}/stream_drone_pid.py" \
  "${BACKUP_DIR}/stream_drone_pid.py"

if [[ -f "${TARGET_DIR}/star_pid.py" ]]; then
  install -o robot3 -g robot3 -m 0644 \
    "${TARGET_DIR}/star_pid.py" "${BACKUP_DIR}/star_pid.py"
fi
for module in star_predictive_pid.py star_realtime.py; do
  if [[ -f "${TARGET_DIR}/${module}" ]]; then
    install -o robot3 -g robot3 -m 0644 \
      "${TARGET_DIR}/${module}" "${BACKUP_DIR}/${module}"
  fi
done
if [[ -f "${TARGET_DIR}/device_auth.py" ]]; then
  install -o robot3 -g robot3 -m 0644 \
    "${TARGET_DIR}/device_auth.py" "${BACKUP_DIR}/device_auth.py"
fi
if [[ -f /etc/star-tracking.env ]]; then
  install -o root -g root -m 0600 \
    /etc/star-tracking.env "${BACKUP_DIR}/star-tracking.env"
fi

ROLLBACK_READY=1
systemctl stop "${SERVICE}"

install -o robot3 -g robot3 -m 0644 \
  "${KIT_DIR}/stream_drone_pid.py" \
  "${TARGET_DIR}/stream_drone_pid.py"
install -o robot3 -g robot3 -m 0644 \
  "${KIT_DIR}/star_predictive_pid.py" \
  "${TARGET_DIR}/star_predictive_pid.py"
install -o robot3 -g robot3 -m 0644 \
  "${KIT_DIR}/star_realtime.py" \
  "${TARGET_DIR}/star_realtime.py"
install -o robot3 -g robot3 -m 0644 \
  "${KIT_DIR}/device_auth.py" \
  "${TARGET_DIR}/device_auth.py"

if [[ ! -f /etc/star-tracking.env ]]; then
  install -o root -g root -m 0600 \
    "${KIT_DIR}/star-tracking.env.example" \
    /etc/star-tracking.env
fi

# Migration des valeurs v1 responsables d'un debit trop eleve sur Jetson Nano.
# Les valeurs personnalisees par l'utilisateur sont conservees.
sed -i 's/^STAR_STREAM_MAX_FPS=10$/STAR_STREAM_MAX_FPS=6/' /etc/star-tracking.env
sed -i 's/^STAR_STREAM_JPEG_QUALITY=65$/STAR_STREAM_JPEG_QUALITY=55/' /etc/star-tracking.env
sed -i 's/^STAR_STREAM_MAX_WIDTH=640$/STAR_STREAM_MAX_WIDTH=512/' /etc/star-tracking.env
sed -i 's/^STAR_STREAM_MAX_HEIGHT=480$/STAR_STREAM_MAX_HEIGHT=384/' /etc/star-tracking.env
grep -q '^STAR_GUET_INFERENCE_HZ=' /etc/star-tracking.env || echo 'STAR_GUET_INFERENCE_HZ=2' >> /etc/star-tracking.env
sed -i 's/^STAR_CANON_INFERENCE_HZ=8$/STAR_CANON_INFERENCE_HZ=0/' /etc/star-tracking.env
sed -i 's/^STAR_GUET_SERVO_HZ=12$/STAR_GUET_SERVO_HZ=20/' /etc/star-tracking.env
grep -q '^STAR_CANON_INFERENCE_HZ=' /etc/star-tracking.env || echo 'STAR_CANON_INFERENCE_HZ=0' >> /etc/star-tracking.env
grep -q '^STAR_GUET_SERVO_HZ=' /etc/star-tracking.env || echo 'STAR_GUET_SERVO_HZ=20' >> /etc/star-tracking.env
grep -q '^STAR_TENSORRT_FP16=' /etc/star-tracking.env || echo 'STAR_TENSORRT_FP16=1' >> /etc/star-tracking.env
grep -q '^STAR_CONTROL_LOOP_HZ=' /etc/star-tracking.env || echo 'STAR_CONTROL_LOOP_HZ=50' >> /etc/star-tracking.env
grep -q '^STAR_GUET_PAN_SCAN_DPS=' /etc/star-tracking.env || echo 'STAR_GUET_PAN_SCAN_DPS=12' >> /etc/star-tracking.env
grep -q '^STAR_GUET_TILT_SCAN_DPS=' /etc/star-tracking.env || echo 'STAR_GUET_TILT_SCAN_DPS=5' >> /etc/star-tracking.env
grep -q '^STAR_HANDOFF_MIN_INTERVAL_S=' /etc/star-tracking.env || echo 'STAR_HANDOFF_MIN_INTERVAL_S=0.15' >> /etc/star-tracking.env
grep -q '^STAR_HANDOFF_MIN_CONFIDENCE=' /etc/star-tracking.env || echo 'STAR_HANDOFF_MIN_CONFIDENCE=0.40' >> /etc/star-tracking.env
grep -q '^STAR_METRICS_PRINT_INTERVAL_S=' /etc/star-tracking.env || echo 'STAR_METRICS_PRINT_INTERVAL_S=2' >> /etc/star-tracking.env
chmod 0600 /etc/star-tracking.env

install -d -o root -g root -m 0755 /etc/systemd/system/drone.service.d
install -o root -g root -m 0644 \
  "${KIT_DIR}/systemd/tracking.conf" \
  /etc/systemd/system/drone.service.d/tracking.conf

sudo -u robot3 env PYTHONPATH=/mnt/sdcard/lib/python3.6/site-packages \
  /usr/bin/python3 -m py_compile \
  "${TARGET_DIR}/star_predictive_pid.py" \
  "${TARGET_DIR}/star_realtime.py" \
  "${TARGET_DIR}/device_auth.py" \
  "${TARGET_DIR}/stream_drone_pid.py"

systemctl daemon-reload
systemctl restart "${SERVICE}"
sleep 3
systemctl --no-pager --full status "${SERVICE}"
ROLLBACK_READY=0
trap - ERR

echo
echo "Installation terminee. Sauvegarde: ${BACKUP_DIR}"
echo "Verifier STAR_ESP32_IP dans /etc/star-tracking.env."
echo "Optimisation temps reel active: latest-frame-only, Canon prioritaire et non plafonne."
echo "Flux normal 6 FPS; Guet IA 2 Hz; commande Canon 30 Hz / controle 50 Hz."
