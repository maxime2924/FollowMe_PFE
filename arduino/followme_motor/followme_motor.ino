// FollowMe — Contrôle moteurs Arduino
// Communication série 115200 bauds depuis Raspberry Pi
// Commandes discretes : AVANCER / STOP / GAUCHE / DROITE / RECULER
// Commande PID (vitesse differentielle continue) : M:gauche,droite  (ex: M:120,-40)

// === PINS (Freenove 4WD shield) ===
#define MOTOR_DIRECTION      0   // Mettre à 1 si les roues tournent à l'envers
#define PIN_DIRECTION_RIGHT  3
#define PIN_DIRECTION_LEFT   4
#define PIN_MOTOR_PWM_RIGHT  5
#define PIN_MOTOR_PWM_LEFT   6

// === ULTRASONS ===
#define PIN_TRIG  7
#define PIN_ECHO  8
#define DISTANCE_STOP 40  // cm — arrêt d'urgence si obstacle < 40cm

// === SERVO RADAR (Tier 3) ===
#include <Servo.h>
#define PIN_SERVO 9
#define SERVO_CENTRE  90   // degres - position avant (arret d'urgence garanti)
#define SERVO_GAUCHE  45
#define SERVO_DROITE  135
#define SERVO_BALAYAGE_INTERVAL 1000  // ms entre deux balayages
#define SERVO_DELAI_MOUVEMENT   150   // ms pour laisser le servo atteindre la position
Servo radarServo;
unsigned long dernierBalayage = 0;

// === VITESSE ===
#define VITESSE_AVANCE  100   // 0-255
#define VITESSE_TOURNE  70
#define PWM_MAX         255   // borne de securite pour la commande M:

void setup() {
  Serial.begin(115200);

  pinMode(PIN_DIRECTION_LEFT,  OUTPUT);
  pinMode(PIN_DIRECTION_RIGHT, OUTPUT);
  pinMode(PIN_MOTOR_PWM_LEFT,  OUTPUT);
  pinMode(PIN_MOTOR_PWM_RIGHT, OUTPUT);

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  radarServo.attach(PIN_SERVO);
  radarServo.write(SERVO_CENTRE);

  motorRun(0, 0);
  Serial.println("Arduino pret. En attente d'ordres...");
}

void loop() {
  // 0. Balayage radar (Tier 3) - non bloquant, base sur millis()
  // Pendant le balayage, le capteur ne regarde pas l'avant : l'arret
  // d'urgence frontal n'est garanti qu'en position SERVO_CENTRE.
  unsigned long maintenant = millis();
  if (maintenant - dernierBalayage >= SERVO_BALAYAGE_INTERVAL) {
    dernierBalayage = maintenant;

    radarServo.write(SERVO_GAUCHE);
    delay(SERVO_DELAI_MOUVEMENT);
    int distGauche = lireDistance();
    if (distGauche > 0) {
      Serial.print("DISTG:");
      Serial.println(distGauche);
    }

    radarServo.write(SERVO_DROITE);
    delay(SERVO_DELAI_MOUVEMENT * 2);  // traversee complete gauche->droite
    int distDroite = lireDistance();
    if (distDroite > 0) {
      Serial.print("DISTD:");
      Serial.println(distDroite);
    }

    radarServo.write(SERVO_CENTRE);
    delay(SERVO_DELAI_MOUVEMENT);
  }

  // 1. Lire distance ultrasons (position courante du servo, normalement CENTRE)
  int distance = lireDistance();

  // 2. Envoyer distance en permanence vers la Pi
  if (distance > 0) {
    Serial.print("DIST:");
    Serial.println(distance);
  }

  // 3. Arrêt d'urgence prioritaire
  if (distance > 0 && distance < DISTANCE_STOP) {
    motorRun(0, 0);
    delay(100);
    return;
  }

  // 4. Lire ordre série
  if (Serial.available() > 0) {
    String ordre = Serial.readStringUntil('\n');
    ordre.trim();

    if (ordre == "AVANCER") {
      motorRun(VITESSE_AVANCE, VITESSE_AVANCE);
      Serial.println("OK: AVANCER");
    }
    else if (ordre == "GAUCHE") {
      motorRun(-VITESSE_TOURNE, VITESSE_TOURNE);
      Serial.println("OK: GAUCHE");
    }
    else if (ordre == "DROITE") {
      motorRun(VITESSE_TOURNE, -VITESSE_TOURNE);
      Serial.println("OK: DROITE");
    }
    else if (ordre == "STOP") {
      motorRun(0, 0);
      Serial.println("OK: STOP");
    }
    else if (ordre == "RECULER") {
      motorRun(-VITESSE_AVANCE, -VITESSE_AVANCE);
      Serial.println("OK: RECULER");
    }
    else if (ordre.startsWith("M:")) {
      // Commande PID : vitesse differentielle continue "M:gauche,droite"
      int sepIndex = ordre.indexOf(',', 2);
      if (sepIndex > 0) {
        int speedL = ordre.substring(2, sepIndex).toInt();
        int speedR = ordre.substring(sepIndex + 1).toInt();

        // Securite : bornage des valeurs reçues
        speedL = constrain(speedL, -PWM_MAX, PWM_MAX);
        speedR = constrain(speedR, -PWM_MAX, PWM_MAX);

        motorRun(speedL, speedR);
        Serial.print("OK: M:");
        Serial.print(speedL);
        Serial.print(",");
        Serial.println(speedR);
      }
    }
  }
}

// === Pilotage moteurs ===
void motorRun(int speedl, int speedr) {
  int dirL = 0, dirR = 0;

  if (speedl > 0) {
    dirL = 0 ^ MOTOR_DIRECTION;
  } else {
    dirL = 1 ^ MOTOR_DIRECTION;
    speedl = -speedl;
  }

  if (speedr > 0) {
    dirR = 1 ^ MOTOR_DIRECTION;
  } else {
    dirR = 0 ^ MOTOR_DIRECTION;
    speedr = -speedr;
  }

  digitalWrite(PIN_DIRECTION_LEFT,  dirL);
  digitalWrite(PIN_DIRECTION_RIGHT, dirR);
  analogWrite(PIN_MOTOR_PWM_LEFT,   speedl);
  analogWrite(PIN_MOTOR_PWM_RIGHT,  speedr);
}

// === Lecture distance HC-SR04 ===
int lireDistance() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long duree = pulseIn(PIN_ECHO, HIGH, 30000);
  if (duree == 0) return -1;
  return duree * 0.034 / 2;
}

