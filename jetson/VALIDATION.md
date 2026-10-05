# Validation S.T.A.R. v3

## Exécuté hors matériel

- compilation syntaxique Python : `star_predictive_pid.py`,
  `star_realtime.py`, `stream_drone_realtime.py` — OK ;
- 13 tests unitaires — OK ;
- couverture : démarrage désarmé, limites, persistance, alpha-bêta, prédiction,
  précision sous le degré, machine d'état, indépendance du contrôle vis-à-vis
  des FPS, priorité Canon et modes de performance ;
- syntaxe Bash des scripts installation/rollback — OK ;
- syntaxe C++11 du firmware avec interfaces Arduino simulées — OK ;
- lint du site — OK ;
- build et validation Worker du site — OK ;
- déploiement Sites version 36 — réussi.

## À exécuter sur le matériel

- compilation PlatformIO avec le véritable framework ESP32 ;
- contrôle des impulsions 500–2500 µs servo par servo ;
- sens de chaque axe et butées mécaniques ;
- FPS/latence réels avec `new_best_v1.engine` sur Jetson Nano ;
- comparaison thermique/froid afin de détecter un throttling ;
- test de perte Wi-Fi et watchdog ;
- test de CAM02 ignorée confirmant que CAM01 continue le tracking.

Les gains de latence ne sont pas chiffrés artificiellement : les valeurs
avant/après doivent venir des lignes `STAR_METRICS` de la Jetson réelle.
