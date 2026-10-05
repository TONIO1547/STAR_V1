# Code Jetson Nano (à ajouter)

Ce dossier accueillera le code qui tourne sur la Jetson Nano :

- détection des drones (YOLOv5n v6.2 optimisé TensorRT, post-traitement NMS) ;
- suivi : erreur de position, filtre passe-bas, PID pan / tilt ;
- envoi des consignes à l'ESP32 en UDP (`canon2:seq,pan,tilt` / `guet2:seq,pan,tilt`) ;
- interface web Flask (flux vidéo annoté, commande des moteurs) ;
- service systemd de démarrage automatique.

Les poids du modèle entraîné (`.pt`, `.engine`, `.onnx`) ne sont pas publiés.
