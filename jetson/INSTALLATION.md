# S.T.A.R. v3 — tracking Canon temps réel

Ce kit cible la Jetson Nano JetPack 4.6.6, l'ESP32 sous PlatformIO, deux
webcams, la caméra thermique UART et le moteur TensorRT
`new_best_v1.engine`. CAM01 (`/video1`, USB 2.1) est toujours la caméra Canon
principale. CAM02 (`/video0`, USB 2.2) sert uniquement à l'acquisition.

## Architecture

```text
CAM01 Canon capture continue → dernière frame uniquement → TensorRT prioritaire
 → alpha-bêta + prédiction → PID vitesse + feed-forward → dernière commande UDP
 → ESP32 séquence + interpolation 100 Hz → writeMicroseconds → servos Canon

CAM02 Guet capture continue → TensorRT opportuniste 2 Hz → transfert immédiat
 → Canon, uniquement pendant SEARCH / REACQUIRE

Flux web → rendu indépendant → JPEG seulement avec spectateur → 6 / 3 / 0 FPS
```

Les threads OpenCV, TensorRT et encodage appellent principalement du code natif
qui libère le GIL. Des processus séparés ajouteraient ici de la copie d'images
et de la complexité sans gain garanti sur la Nano.

## Fichiers principaux

- `stream_drone_pid.py` : service complet, installé sous ce même nom dans
  `/mnt/sdcard/yolov5` ;
- `star_realtime.py` : capture/latest-only, priorité GPU, streaming et
  sortie UDP latest-only ;
- `star_predictive_pid.py` : estimation, prédiction, états et contrôle ;
- `../firmware_esp32/src/main.cpp` : firmware complet haute résolution ;
- `scripts/install_jetson.sh` : sauvegarde, installation, validation et
  redémarrage ;
- `scripts/rollback_jetson.sh` : restauration de l'ancien service et de son
  environnement ;

## Sécurité électrique et mécanique

Alimenter les servos par une alimentation 5–6 V séparée, dimensionnée pour leur
courant de blocage. Relier le GND de cette alimentation au GND ESP32. Ne pas
alimenter quatre servos depuis l'ESP32. Les GPIO 5, 16, 18 et 19 sont uniquement
des signaux.

Les limites sont vérifiées sur le site, la Jetson et l'ESP32 : PAN 0–180°,
TILT 90–150°. Le laser GPIO 17 reste forcé à OFF, y compris si un paquet
`laser_on` est reçu. Les moteurs ont un watchdog de 350 ms et les trajectoires
servo un watchdog de 1,2 s.

## 1. ESP32 — VS Code + PlatformIO

1. Ouvrir le dossier `../firmware_esp32`.
2. Copier `include/secrets.example.h` vers `include/secrets.h`.
3. Mettre le SSID et le mot de passe du point d’accès dans la copie locale
   `include/secrets.h`. Ce fichier n’est jamais versionné.
4. Garder `STAR_USE_STATIC_IP 0` pour le premier démarrage.
5. Compiler, téléverser, puis ouvrir le moniteur série à 115200 bauds.
6. Relever `ESP32 pret, IP:` : c'est la valeur à mettre dans
   `STAR_ESP32_IP` (notée `<IP_ESP32>` dans la suite de ce document).

Commandes PlatformIO équivalentes :

```bash
cd ../firmware_esp32
pio run
pio run -t upload
pio device monitor -b 115200
```

Le protocole moderne est `canon2:SEQ,PAN,TILT` / `guet2:SEQ,PAN,TILT`. Les
anciens paquets entiers restent acceptés pour permettre un rollback Jetson.

## 2. Jetson — installation

Copier ce dépôt sur la Jetson, puis lancer :

```bash
cd jetson
chmod +x scripts/install_jetson.sh scripts/rollback_jetson.sh
sudo bash scripts/install_jetson.sh
```

Le script ne remplace pas un `/etc/star-tracking.env` déjà présent. Vérifier
donc ensuite :

```bash
sudo sed -i 's/^STAR_ESP32_IP=.*/STAR_ESP32_IP=<IP_ESP32>/' /etc/star-tracking.env
sudo grep '^STAR_ESP32_' /etc/star-tracking.env
sudo systemctl daemon-reload
sudo systemctl restart drone.service
sudo systemctl status drone.service --no-pager -l
```

Valeurs attendues : Canon non plafonné (`STAR_CANON_INFERENCE_HZ=0`), Guet
2 Hz, contrôle 50 Hz et commandes PID 30 Hz.

## 3. Vérification et métriques

```bash
sudo journalctl -u drone.service -f
```

Une ligne `STAR_METRICS` apparaît toutes les deux secondes. Elle contient FPS
capture/IA, preprocessing, TensorRT, NMS, tracking, âge de frame, latence
frame→commande, fréquence servo et coût JPEG.

Tester les API sans imprimer le secret :

```bash
sudo bash -c '. /etc/star-device.env; curl -s \
  -H "Authorization: Bearer $STAR_DEVICE_TOKEN" \
  http://127.0.0.1:5000/api/performance'

sudo bash -c '. /etc/star-device.env; curl -s \
  -H "Authorization: Bearer $STAR_DEVICE_TOKEN" \
  http://127.0.0.1:5000/api/pid'

curl -s -o /dev/null -w '%{http_code}\n' \
  http://127.0.0.1:5000/api/pid
```

Les deux premières réponses doivent contenir `"ok":true`; la dernière doit
renvoyer `401`.

## 4. Site et modes de performance

Le panneau publié sur `https://star-ai.fr` propose :

| Mode | Tracking Canon | Flux web | Conséquence |
|---|---|---|---|
| Normal | priorité maximale | 6 FPS, 512×384, JPEG 55 | meilleur confort visuel |
| Performance | priorité maximale | 3 FPS, 384×288, JPEG 45 | charge CPU/JPEG réduite |
| Maximum | priorité absolue | coupé | aucune vidéo live, tracking actif |

Le bouton CAM02 arrête son IA et son balayage sans désarmer CAM01. Le flux
CAM02 reste visible afin que l'opérateur puisse vérifier la scène.

## 5. Réglage du suivi

Commencer dans une zone libre, laser physiquement déconnecté et vitesse basse.
Ne modifier le PID qu'après avoir vérifié que `frame_age_ms` reste faible.

1. Armer CAM01 depuis le site.
2. Vérifier le sens PAN/TILT ; désarmer immédiatement si un axe s'éloigne.
3. Régler Kp avec Ki et Kd faibles.
4. Ajouter Kd pour amortir.
5. Garder Ki minimal ; l'anti-windup bloque son accumulation en saturation.
6. Régler ensuite le feed-forward selon les mouvements rapides.
7. Augmenter vitesse/accélération seulement après validation mécanique.

Départs prudents : alpha 0,70, bêta 0,15, prédiction 40 ms limitée à 80 px,
PAN 35°/s et TILT 25°/s. Les valeurs sont appliquées en direct ;
`Appliquer + mémoriser` sauvegarde les réglages, mais un reboot reste toujours
désarmé.

## 6. Comparaison avant/après

1. Sauvegarder 30 s de lignes `STAR_METRICS` sans site ouvert.
2. Ouvrir CAM01 pendant 30 s.
3. Ouvrir les trois flux pendant 30 s.
4. Passer en Performance puis Maximum, 30 s chacun.
5. Comparer `inference_hz`, `inference_ms`, `frame_age_ms`,
   `frame_to_command_ms`, `servo_command_hz` et `jpeg_ms`.
6. Ne comparer qu'avec la même scène, le même modèle, le même seuil et la même
   température Jetson.

La valeur avant modification doit être enregistrée depuis l'ancien service ;
le kit ne fabrique aucune mesure rétroactive.

## 7. Résolution et tracker léger

Le moteur TensorRT actuel est construit pour 640×640. Cette résolution est
conservée car les drones lointains sont petits. Passer à 512 ou 416 exige de
reconstruire et valider un moteur ; le gain de latence peut être annulé par la
perte de rappel. Une ROI fixe n'accélère pas un moteur qui reçoit toujours
640×640 et augmente le risque de perdre un drone rapide.

Le suivi entre inférences est assuré par l'estimateur alpha-bêta, la prédiction
timestampée et la boucle de contrôle 50 Hz. Un CSRT/MOSSE CPU n'est pas activé
par défaut sur la Nano : sa charge et sa dérive peuvent coûter plus que le gain.

## 8. Retour arrière

Le chemin de sauvegarde est affiché après installation :

```bash
sudo bash scripts/rollback_jetson.sh \
  /mnt/sdcard/yolov5/backups/star-pid-AAAAMMJJ-HHMMSS
```

La restauration remet aussi `/etc/star-tracking.env`, puis recharge systemd.
