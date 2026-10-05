#include <Arduino.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_system.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include "secrets.h"

namespace {

// ============================================================================
// PARAMETRES TEMPS REEL — point unique de reglage ESP32
// ============================================================================
constexpr uint16_t UDP_PORT = 4210;
constexpr size_t PACKET_CAPACITY = 160;
constexpr uint32_t MOTOR_WATCHDOG_MS = 350;
constexpr uint32_t SERVO_WATCHDOG_MS = 1200;
constexpr uint32_t SERVO_UPDATE_MS = 10;  // interpolation interne a 100 Hz
// Un partage de connexion de telephone peut mettre plus de 5 s a associer :
// ne pas interrompre une tentative en cours trop tot.
constexpr uint32_t WIFI_RETRY_MS = 15000;

// Limites mecaniques finales. Elles dupliquent volontairement la barriere
// Jetson : un paquet UDP ou une commande web ne peut jamais les depasser.
constexpr float PAN_MIN_DEG = 0.0F;
constexpr float PAN_MAX_DEG = 180.0F;
constexpr float TILT_MIN_DEG = 90.0F;
constexpr float TILT_MAX_DEG = 150.0F;
constexpr float PAN_CENTER_DEG = 90.0F;
constexpr float TILT_CENTER_DEG = 120.0F;

// Calibration des impulsions correspondant a 0 et 180 degres. Commencer avec
// ces valeurs equivalentes a l'ancien attach(500, 2500), puis calibrer chaque
// axe avec la mecanique deconnectee si le modele de servo l'exige.
constexpr int PAN0_US_AT_0 = 500;
constexpr int PAN0_US_AT_180 = 2500;
constexpr int TILT0_US_AT_0 = 500;
constexpr int TILT0_US_AT_180 = 2500;
constexpr int PAN1_US_AT_0 = 500;
constexpr int PAN1_US_AT_180 = 2500;
constexpr int TILT1_US_AT_0 = 500;
constexpr int TILT1_US_AT_180 = 2500;

// Ces limites ne doivent pas etre inferieures aux vitesses imposees par la
// Jetson, sinon l'interpolation ajouterait du retard. Elles servent seulement
// a lisser une grande commande manuelle ou un recentrage.
// Abaissees (240 / 180 auparavant) : un mouvement plus lent limite les pics
// de courant des servos qui provoquaient des resets brownout de l'ESP32.
// Reste au-dessus des vitesses max du PID Jetson (35 / 25 deg/s).
constexpr float CANON_MAX_INTERPOLATION_DPS = 120.0F;
constexpr float GUET_MAX_INTERPOLATION_DPS = 90.0F;

// Canon / CAM01 principale. Pan et tilt inverses par rapport au cablage
// d'origine (19 = pan, 16 = tilt) pour correspondre au montage actuel.
constexpr int PIN_PAN1 = 16;
constexpr int PIN_TILT1 = 19;
constexpr int PIN_LASER = 17;

// Guet / CAM02 acquisition.
constexpr int PIN_PAN0 = 5;
constexpr int PIN_TILT0 = 18;

// Moteurs de deplacement.
constexpr int PIN_PWM1 = 26;
constexpr int PIN_DIR1 = 27;
constexpr int PIN_PWM2 = 33;
constexpr int PIN_DIR2 = 25;
constexpr int CH_PWM1 = 0;
constexpr int CH_PWM2 = 1;

struct AxisState {
  Servo *servo;
  float currentDeg;
  float targetDeg;
  float minimumDeg;
  float maximumDeg;
  float maxRateDps;
  int pulseAt0;
  int pulseAt180;
  uint32_t lastCommandMs;
};

WiFiUDP udp;
Servo servoPan0;
Servo servoTilt0;
Servo servoPan1;
Servo servoTilt1;

char packet[PACKET_CAPACITY];
bool udpStarted = false;
bool motorsActive = false;
uint32_t lastMotorCommandMs = 0;
uint32_t lastWifiAttemptMs = 0;
uint32_t lastServoUpdateMs = 0;
int motorLeft = 0;
int motorRight = 0;

AxisState pan0 = {&servoPan0, PAN_CENTER_DEG, PAN_CENTER_DEG,
                  PAN_MIN_DEG, PAN_MAX_DEG, GUET_MAX_INTERPOLATION_DPS,
                  PAN0_US_AT_0, PAN0_US_AT_180, 0};
AxisState tilt0 = {&servoTilt0, TILT_CENTER_DEG, TILT_CENTER_DEG,
                   TILT_MIN_DEG, TILT_MAX_DEG, GUET_MAX_INTERPOLATION_DPS,
                   TILT0_US_AT_0, TILT0_US_AT_180, 0};
AxisState pan1 = {&servoPan1, PAN_CENTER_DEG, PAN_CENTER_DEG,
                  PAN_MIN_DEG, PAN_MAX_DEG, CANON_MAX_INTERPOLATION_DPS,
                  PAN1_US_AT_0, PAN1_US_AT_180, 0};
AxisState tilt1 = {&servoTilt1, TILT_CENTER_DEG, TILT_CENTER_DEG,
                   TILT_MIN_DEG, TILT_MAX_DEG, CANON_MAX_INTERPOLATION_DPS,
                   TILT1_US_AT_0, TILT1_US_AT_180, 0};

uint32_t lastCanonSequence = 0;
uint32_t lastGuetSequence = 0;
bool hasCanonSequence = false;
bool hasGuetSequence = false;

float clampAngle(float value, float minimum, float maximum) {
  return fmaxf(minimum, fminf(maximum, value));
}

int angleToPulse(const AxisState &axis, float angleDeg) {
  const float safe = clampAngle(angleDeg, axis.minimumDeg, axis.maximumDeg);
  const float ratio = safe / 180.0F;
  const float pulse = axis.pulseAt0 + ratio * (axis.pulseAt180 - axis.pulseAt0);
  return static_cast<int>(lroundf(pulse));
}

void writeAxis(AxisState &axis) {
  axis.currentDeg = clampAngle(axis.currentDeg, axis.minimumDeg, axis.maximumDeg);
  axis.servo->writeMicroseconds(angleToPulse(axis, axis.currentDeg));
}

void setTarget(AxisState &axis, float targetDeg, uint32_t commandMs) {
  axis.targetDeg = clampAngle(targetDeg, axis.minimumDeg, axis.maximumDeg);
  axis.lastCommandMs = commandMs;
}

void holdServos() {
  pan0.targetDeg = pan0.currentDeg;
  tilt0.targetDeg = tilt0.currentDeg;
  pan1.targetDeg = pan1.currentDeg;
  tilt1.targetDeg = tilt1.currentDeg;
}

void setGuetTarget(float panDeg, float tiltDeg, uint32_t commandMs) {
  setTarget(pan0, panDeg, commandMs);
  setTarget(tilt0, tiltDeg, commandMs);
}

void setCanonTarget(float panDeg, float tiltDeg, uint32_t commandMs) {
  setTarget(pan1, panDeg, commandMs);
  setTarget(tilt1, tiltDeg, commandMs);
}

void updateAxis(AxisState &axis, float dtSeconds, uint32_t nowMs) {
  if (nowMs - axis.lastCommandMs > SERVO_WATCHDOG_MS) {
    axis.targetDeg = axis.currentDeg;
  }
  const float error = axis.targetDeg - axis.currentDeg;
  const float maxStep = axis.maxRateDps * dtSeconds;
  if (fabsf(error) <= maxStep) {
    axis.currentDeg = axis.targetDeg;
  } else {
    axis.currentDeg += copysignf(maxStep, error);
  }
  writeAxis(axis);
}

void updateServos() {
  const uint32_t now = millis();
  const uint32_t elapsedMs = now - lastServoUpdateMs;
  if (elapsedMs < SERVO_UPDATE_MS) return;
  lastServoUpdateMs = now;
  const float dt = fminf(static_cast<float>(elapsedMs) / 1000.0F, 0.05F);
  updateAxis(pan0, dt, now);
  updateAxis(tilt0, dt, now);
  updateAxis(pan1, dt, now);
  updateAxis(tilt1, dt, now);
}

void setMotor1(int value) {
  value = constrain(value, -255, 255);
  if (value == 0) {
    ledcWrite(CH_PWM1, 0);
    digitalWrite(PIN_DIR1, LOW);
  } else if (value > 0) {
    digitalWrite(PIN_DIR1, HIGH);
    ledcWrite(CH_PWM1, value);
  } else {
    digitalWrite(PIN_DIR1, LOW);
    ledcWrite(CH_PWM1, -value);
  }
}

void setMotor2(int value) {
  value = constrain(value, -255, 255);
  if (value == 0) {
    ledcWrite(CH_PWM2, 0);
    digitalWrite(PIN_DIR2, LOW);
  } else if (value > 0) {
    digitalWrite(PIN_DIR2, LOW);
    ledcWrite(CH_PWM2, value);
  } else {
    digitalWrite(PIN_DIR2, HIGH);
    ledcWrite(CH_PWM2, -value);
  }
}

void stopMotors() {
  motorLeft = 0;
  motorRight = 0;
  motorsActive = false;
  setMotor1(0);
  setMotor2(0);
}

void safeOutputs() {
  stopMotors();
  digitalWrite(PIN_LASER, LOW);
  holdServos();
}

bool sequenceIsNewer(uint32_t sequence, uint32_t previous, bool hasPrevious) {
  if (!hasPrevious) return true;
  return static_cast<int32_t>(sequence - previous) > 0;
}

void sendStatus(const IPAddress &address, uint16_t port) {
  char reply[280];
  // reset=15 (ESP_RST_BROWNOUT) signale une chute de tension de l'alimentation.
  const int length = snprintf(
      reply, sizeof(reply),
      "status:wifi=%d,pan0=%.3f,tilt0=%.3f,pan1=%.3f,tilt1=%.3f,left=%d,right=%d,seq0=%lu,seq1=%lu,uptime_s=%lu,reset=%d",
      WiFi.status() == WL_CONNECTED ? 1 : 0,
      pan0.currentDeg, tilt0.currentDeg, pan1.currentDeg, tilt1.currentDeg,
      motorLeft, motorRight,
      static_cast<unsigned long>(lastGuetSequence),
      static_cast<unsigned long>(lastCanonSequence),
      static_cast<unsigned long>(millis() / 1000UL),
      static_cast<int>(esp_reset_reason()));
  if (length <= 0) return;
  udp.beginPacket(address, port);
  udp.write(reinterpret_cast<const uint8_t *>(reply),
            static_cast<size_t>(min(length, static_cast<int>(sizeof(reply) - 1))));
  udp.endPacket();
}

void processPacket(const char *message, const IPAddress &remoteAddress,
                   uint16_t remotePort) {
  const uint32_t now = millis();
  if (strcmp(message, "laser_off") == 0 || strcmp(message, "laser_on") == 0) {
    // Le laser ne peut jamais etre active par le tracking visuel ou le reseau.
    digitalWrite(PIN_LASER, LOW);
    return;
  }
  if (strcmp(message, "stop") == 0) {
    safeOutputs();
    return;
  }
  if (strcmp(message, "center") == 0) {
    safeOutputs();
    setGuetTarget(PAN_CENTER_DEG, TILT_CENTER_DEG, now);
    setCanonTarget(PAN_CENTER_DEG, TILT_CENTER_DEG, now);
    return;
  }
  if (strcmp(message, "status?") == 0) {
    sendStatus(remoteAddress, remotePort);
    return;
  }

  unsigned long sequenceRaw = 0;
  float panDeg = 0.0F;
  float tiltDeg = 0.0F;
  if (sscanf(message, "canon2:%lu,%f,%f", &sequenceRaw, &panDeg, &tiltDeg) == 3) {
    const uint32_t sequence = static_cast<uint32_t>(sequenceRaw);
    if (isfinite(panDeg) && isfinite(tiltDeg) &&
        sequenceIsNewer(sequence, lastCanonSequence, hasCanonSequence)) {
      lastCanonSequence = sequence;
      hasCanonSequence = true;
      setCanonTarget(panDeg, tiltDeg, now);
    }
    return;
  }
  if (sscanf(message, "guet2:%lu,%f,%f", &sequenceRaw, &panDeg, &tiltDeg) == 3) {
    const uint32_t sequence = static_cast<uint32_t>(sequenceRaw);
    if (isfinite(panDeg) && isfinite(tiltDeg) &&
        sequenceIsNewer(sequence, lastGuetSequence, hasGuetSequence)) {
      lastGuetSequence = sequence;
      hasGuetSequence = true;
      setGuetTarget(panDeg, tiltDeg, now);
    }
    return;
  }

  // Compatibilite avec l'ancien service Jetson pendant un rollback.
  int legacyPan = 0;
  int legacyTilt = 0;
  if (sscanf(message, "guet:%d,%d", &legacyPan, &legacyTilt) == 2) {
    setGuetTarget(static_cast<float>(legacyPan), static_cast<float>(legacyTilt), now);
    return;
  }
  if (sscanf(message, "canon:%d,%d", &legacyPan, &legacyTilt) == 2) {
    setCanonTarget(static_cast<float>(legacyPan), static_cast<float>(legacyTilt), now);
    return;
  }

  int left = 0;
  int right = 0;
  if (sscanf(message, "motor:%d,%d", &left, &right) == 2) {
    motorLeft = constrain(left, -255, 255);
    motorRight = constrain(right, -255, 255);
    setMotor1(motorLeft);
    setMotor2(motorRight);
    lastMotorCommandMs = now;
    motorsActive = motorLeft != 0 || motorRight != 0;
  }
}

void configureWifi() {
#if STAR_USE_STATIC_IP
  const IPAddress local(STAR_STATIC_IP_1, STAR_STATIC_IP_2, STAR_STATIC_IP_3,
                        STAR_STATIC_IP_4);
  const IPAddress gateway(STAR_GATEWAY_IP_1, STAR_GATEWAY_IP_2,
                          STAR_GATEWAY_IP_3, STAR_GATEWAY_IP_4);
  const IPAddress subnet(STAR_SUBNET_1, STAR_SUBNET_2, STAR_SUBNET_3,
                         STAR_SUBNET_4);
  if (!WiFi.config(local, gateway, subnet)) {
    Serial.println("Configuration IP statique refusee");
  }
#endif
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // evite la latence ajoutee par l'economie d'energie
  WiFi.begin(STAR_WIFI_SSID, STAR_WIFI_PASSWORD);
  lastWifiAttemptMs = millis();
}

void maintainWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!udpStarted) {
      udpStarted = udp.begin(UDP_PORT) == 1;
      if (udpStarted) {
        Serial.print("ESP32 pret, IP: ");
        Serial.println(WiFi.localIP());
        Serial.print("UDP: ");
        Serial.println(UDP_PORT);
      }
    }
    return;
  }
  if (udpStarted) {
    udp.stop();
    udpStarted = false;
  }
  safeOutputs();
  const uint32_t now = millis();
  if (now - lastWifiAttemptMs >= WIFI_RETRY_MS) {
    lastWifiAttemptMs = now;
    // 1 = reseau introuvable (eteint ou 5 GHz), 4 = mot de passe refuse,
    // 6 = deconnecte / pas de reponse du point d'acces.
    const int status = static_cast<int>(WiFi.status());
    Serial.printf("Reconnexion WiFi... (statut %d)\n", status);
    const int found = WiFi.scanNetworks();
    bool seen = false;
    for (int i = 0; i < found; ++i) {
      if (WiFi.SSID(i) == STAR_WIFI_SSID) {
        seen = true;
        Serial.printf("  '%s' visible, signal %d dBm, canal %d\n",
                      STAR_WIFI_SSID, WiFi.RSSI(i), WiFi.channel(i));
      }
    }
    if (!seen) {
      Serial.printf("  '%s' INVISIBLE (%d reseaux 2,4 GHz vus)\n",
                    STAR_WIFI_SSID, found);
    }
    WiFi.scanDelete();
    WiFi.disconnect();
    WiFi.begin(STAR_WIFI_SSID, STAR_WIFI_PASSWORD);
  }
}

void readUdpLatest() {
  if (!udpStarted) return;
  // Toute la file UDP est drainee avant updateServos(). Si plusieurs consignes
  // sont arrivees, seule la cible la plus recente produit un mouvement.
  for (;;) {
    const int packetSize = udp.parsePacket();
    if (packetSize <= 0) return;
    const int length = udp.read(packet, sizeof(packet) - 1);
    if (length <= 0) continue;
    packet[length] = '\0';
    processPacket(packet, udp.remoteIP(), udp.remotePort());
  }
}

}  // namespace

void setup() {
  Serial.begin(115200);
  Serial.print("Demarrage ESP32, cause du reset: ");
  Serial.println(static_cast<int>(esp_reset_reason()));
  pinMode(PIN_LASER, OUTPUT);
  pinMode(PIN_DIR1, OUTPUT);
  pinMode(PIN_DIR2, OUTPUT);

  ledcSetup(CH_PWM1, 1000, 8);
  ledcAttachPin(PIN_PWM1, CH_PWM1);
  ledcSetup(CH_PWM2, 1000, 8);
  ledcAttachPin(PIN_PWM2, CH_PWM2);

  servoPan1.setPeriodHertz(50);
  servoTilt1.setPeriodHertz(50);
  servoPan0.setPeriodHertz(50);
  servoTilt0.setPeriodHertz(50);
  // Demarrage echelonne : un servo a la fois rejoint le centre, pour eviter
  // que les quatre appels de courant simultanes declenchent le brownout.
  AxisState *axes[] = {&pan1, &tilt1, &pan0, &tilt0};
  const int pins[] = {PIN_PAN1, PIN_TILT1, PIN_PAN0, PIN_TILT0};
  for (int i = 0; i < 4; ++i) {
    axes[i]->servo->attach(pins[i], axes[i]->pulseAt0, axes[i]->pulseAt180);
    writeAxis(*axes[i]);
    delay(400);
  }

  const uint32_t now = millis();
  pan0.lastCommandMs = tilt0.lastCommandMs = now;
  pan1.lastCommandMs = tilt1.lastCommandMs = now;
  safeOutputs();
  configureWifi();
  // Puissance d'emission reduite : moins de pics de courant radio, largement
  // suffisante a quelques metres du telephone.
  WiFi.setTxPower(WIFI_POWER_15dBm);
}

void loop() {
  maintainWifi();
  readUdpLatest();
  updateServos();

  if (motorsActive && millis() - lastMotorCommandMs > MOTOR_WATCHDOG_MS) {
    stopMotors();
  }
  delay(1);
}
