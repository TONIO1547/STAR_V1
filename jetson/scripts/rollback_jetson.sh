#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Lancer avec sudo et fournir le dossier de sauvegarde."
  exit 1
fi

BACKUP_DIR="${1:-}"
TARGET_DIR="/mnt/sdcard/yolov5"

case "${BACKUP_DIR}" in
  "${TARGET_DIR}"/backups/star-pid-*) ;;
  *)
    echo "Dossier de sauvegarde refuse: ${BACKUP_DIR}"
    exit 1
    ;;
esac

if [[ ! -f "${BACKUP_DIR}/stream_drone_pid.py" ]]; then
  echo "Sauvegarde invalide: stream_drone_pid.py absent"
  exit 1
fi

systemctl stop drone.service
install -o robot3 -g robot3 -m 0644 \
  "${BACKUP_DIR}/stream_drone_pid.py" "${TARGET_DIR}/stream_drone_pid.py"

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

systemctl daemon-reload
systemctl restart drone.service
systemctl --no-pager --full status drone.service
