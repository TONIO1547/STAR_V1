#pragma once

// Copier ce fichier en secrets.h et renseigner le WiFi auquel la Jetson est
// connectee. secrets.h n'est jamais versionne.
#define STAR_WIFI_SSID "mon-reseau"
#define STAR_WIFI_PASSWORD "mon-mot-de-passe"

// DHCP conseille : l'IP obtenue est affichee dans le moniteur serie et
// l'ESP32 repond au nom mDNS star-esp32.local.
#define STAR_USE_STATIC_IP 0

#define STAR_STATIC_IP_1 10
#define STAR_STATIC_IP_2 206
#define STAR_STATIC_IP_3 166
#define STAR_STATIC_IP_4 92

#define STAR_GATEWAY_IP_1 10
#define STAR_GATEWAY_IP_2 206
#define STAR_GATEWAY_IP_3 166
#define STAR_GATEWAY_IP_4 1

#define STAR_SUBNET_1 255
#define STAR_SUBNET_2 255
#define STAR_SUBNET_3 255
#define STAR_SUBNET_4 0
