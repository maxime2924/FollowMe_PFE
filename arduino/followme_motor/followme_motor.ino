// FollowMe — Contrôle moteurs Arduino
// Communication série 115200 bauds depuis Raspberry Pi
// Commandes : AVANCER / STOP / GAUCHE / DROITE / RECULER

// === PINS (Freenove 4WD shield) ===
#define MOTOR_DIRECTION      0   // Mettre à 1 si les roues tournent à l'envers
#define PIN_DIRECTION_RIGHT  3
#define PIN_DIRECTION_LEFT   4
#define PIN_MOTOR_PWM_RIGHT  5
#define PIN_MOTOR_PWM_LEFT   6

// === ULTRASONS ===
#define PIN_TRIG  7
#define PIN_ECHO  8
#define DISTANCE_STOP 20  // cm — arrêt d'urgence si obstacle < 20cm

// === VITESSE ===
#define VITESSE_AVANCE  100   // 0-255
#define VITESSE_TOURNE  70

void setup() {
  Serial.begin(115200);

  pinMode(PIN_DIRECTION_LEFT,  OUTPUT);
  pinMode(PIN_DIRECTION_RIGHT, OUTPUT);
  pinMode(PIN_MOTOR_PWM_LEFT,  OUTPUT);
  pinMode(PIN_MOTOR_PWM_RIGHT, OUTPUT);

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);

  motorRun(0, 0);
  Serial.println("Arduino pret. En attente d'ordres...");
}

void loop() {
  // 1. Lire distance ultrasons
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