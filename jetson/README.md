# Code Jetson Nano

Code tel qu'il tourne sur la Jetson Nano du prototype V1 (JetPack 4.6.6,
Python 3.6, TensorRT). Le service `drone.service` lance
[`stream_drone_pid.py`](stream_drone_pid.py) depuis `/mnt/sdcard/yolov5` : la
carte SD héberge PyTorch, TorchVision et YOLOv5 v6.2 via `PYTHONUSERBASE`,
faute de place sur la partition système.

> Les poids du modèle (`.pt`, `.engine`, `.onnx`) ne sont pas publiés, et les
> fichiers d'environnement réels (`/etc/star.env`, `/etc/star-device.env`,
> `/etc/star-tracking.env`) non plus : seuls les modèles `*.env.example` le sont.

## Chaîne de traitement

```text
CAM01 Canon  capture continue → dernière frame seulement → TensorRT prioritaire
  → alpha-bêta + prédiction → PID vitesse + feed-forward → dernière commande UDP
  → ESP32 (séquence, interpolation 100 Hz, writeMicroseconds) → servos canon

CAM02 Guet   capture continue → TensorRT opportuniste 2 Hz → handoff immédiat
  → canon, uniquement en état SEARCH / REACQUIRE

Flux web     rendu indépendant → JPEG uniquement s'il y a un spectateur
  → 6 / 3 / 0 FPS selon le mode de performance
```

Aucune file d'images ni de commandes ne s'accumule : chaque étage ne garde que
la donnée la plus récente, pour que la latence ne dérive pas quand la Nano
sature.

## Fichiers

| Fichier | Rôle |
|---|---|
| [`stream_drone_pid.py`](stream_drone_pid.py) | service complet : caméras, inférence TensorRT, NMS maison, PID, UDP vers l'ESP32, API et flux Flask |
| [`star_realtime.py`](star_realtime.py) | briques temps réel : `LatestFrameCamera`, `PriorityInferenceGate`, `PerformanceMode`, `StreamHub`, `LatestServoSender`, métriques |
| [`star_predictive_pid.py`](star_predictive_pid.py) | `AlphaBetaEstimator`, `VelocityPID` (anti-windup, limites de vitesse et d'accélération) et `TrackingController` (machine d'état SEARCH / TRACK / COAST / REACQUIRE) |
| [`star_pid.py`](star_pid.py) | PID position + filtre passe-bas de la première version du suivi, conservé pour le retour arrière |
| [`device_auth.py`](device_auth.py) | jeton `Bearer` exigé sur toutes les routes Flask |
| [`star_auth.py`](star_auth.py) | session et mot de passe de l'interface (PBKDF2-SHA256, 600 000 itérations, verrouillage après 5 échecs) |
| [`star_runtime_api.py`](star_runtime_api.py) | API seuil de confiance et température du SoC |
| [`star_detection_report.py`](star_detection_report.py) | historique des détections avec capture annotée, purge à 24 h |
| [`star_pid_config.json`](star_pid_config.json) | gains réellement utilisés sur le robot |
| [`systemd/`](systemd) | unité `drone.service` et son drop-in `tracking.conf` |
| [`scripts/`](scripts) | installation avec sauvegarde/restauration automatique, et rollback |
| [`tests/`](tests) | 13 tests unitaires du suivi, sans matériel |
| [`legacy/`](legacy) | premiers scripts de détection et de diffusion, avant le PID |

## Protocole UDP vers l'ESP32

Port 4210. Un numéro de séquence permet à l'ESP32 d'ignorer un paquet arrivé en
retard, et les angles sont envoyés en flottants (0,001° de résolution) :

```text
canon2:SEQ,PAN,TILT      guet2:SEQ,PAN,TILT
motor:GAUCHE,DROITE      stop        center      status?
```

## API Flask

Toutes les routes exigent l'en-tête `Authorization: Bearer $STAR_DEVICE_TOKEN`.

| Route | Usage |
|---|---|
| `/video0`, `/video1`, `/thermal`, `/fusion` | flux MJPEG |
| `/api/pid` | lecture et réglage des gains, armement du suivi |
| `/api/servos` | commande manuelle des deux tourelles, `center`, `stop` |
| `/api/performance` | mode Normal / Performance / Maximum |
| `/api/confidence` | seuil de détection |
| `/api/jetson-temperature` | température du SoC |
| `/api/detections` | historique des détections |
| `/motor`, `/status` | base à chenilles, état courant |

Le suivi démarre toujours **désarmé**, y compris après un redémarrage, et une
commande manuelle du canon désarme le suivi et le laser.

## Configuration

Trois fichiers d'environnement, tous en `root:root` mode `0600` :

| Fichier | Modèle | Contenu |
|---|---|---|
| `/etc/star-tracking.env` | [`star-tracking.env.example`](star-tracking.env.example) | caméras, IP de l'ESP32, cadences, flux, fusion thermique |
| `/etc/star-device.env` | [`star-device.env.example`](star-device.env.example) | jeton partagé avec le site |
| `/etc/star.env` | [`star.env.example`](star.env.example) | identifiant et hachage du mot de passe de l'interface |

Deux variables ne figurent dans aucun modèle car elles sont facultatives :
`STAR_CONFIDENCE` (seuil initial, 0,25 par défaut) et `STAR_THERMAL_PORT` (port
UART de la caméra thermique ; sans elle, les flux `/thermal` et `/fusion`
restent vides).

## Installer, vérifier, revenir en arrière

```bash
cd jetson
chmod +x scripts/install_jetson.sh scripts/rollback_jetson.sh
sudo bash scripts/install_jetson.sh
```

Le script sauvegarde la version en place dans
`/mnt/sdcard/yolov5/backups/star-pid-AAAAMMJJ-HHMMSS/`, installe, compile,
redémarre le service, et restaure tout seul si une étape échoue. Pour revenir
en arrière plus tard :

```bash
sudo bash scripts/rollback_jetson.sh /mnt/sdcard/yolov5/backups/star-pid-AAAAMMJJ-HHMMSS
```

Les tests tournent sans matériel ni caméra (`cv2` est remplacé par un module
vide s'il est absent) :

```bash
cd jetson && python3 tests/test_star_pid.py -v
```

Le détail de l'installation, du réglage du PID et de la lecture des métriques
`STAR_METRICS` est dans [`INSTALLATION.md`](INSTALLATION.md), et l'état de la
validation dans [`VALIDATION.md`](VALIDATION.md).
