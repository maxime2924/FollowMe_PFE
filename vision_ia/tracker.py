import cv2
import numpy as np
import onnxruntime as ort
import serial
import time
import threading
from flask import Flask, Response
from picamera2 import Picamera2

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False

app = Flask(__name__)
tracker_instance = None
running = True

# === SEUILS MACHINE A ETATS (alignes sur architecture_compagnon / pfe_historique) ===
# ERREUR_X_FORTE et ERREUR_X_DOUCE ne sont plus utilises directement (remplaces
# par le PID continu ci-dessous) mais conserves en reference / historique.
ERREUR_X_FORTE = 50      # px - ancien seuil rotation forte (pre-PID)
ERREUR_X_DOUCE = 40      # px - ancien seuil correction douce (pre-PID)
DIST_TROP_PRES = 45      # cm - en-dessous: passage TROP_PROCHE
DIST_RETOUR_SUIVRE = 60  # cm - au-dessus (depuis TROP_PROCHE): retour SUIVRE
DIST_AVANCER = 80        # cm - au-dessus: AVANCER
RECUL_TIMEOUT = 2.0      # secondes - securite anti-mur en TROP_PROCHE
SAUT_OF_MAX = 100        # px - au-dela: saut impossible OF -> cible perdue forcee

# === EF_04 - Signal de detresse (cible perdue) ===
PIN_BUZZER = 17    # GPIO17 (pin physique 11)
PIN_LED_R = 22     # GPIO22 (pin physique 15)
PIN_LED_G = 23     # GPIO23 (pin physique 16)
PIN_LED_B = 24     # GPIO24 (pin physique 18)
PERDU_DELAI_SIGNAL = 3.0   # secondes en PERDU avant declenchement signal
SIGNAL_PERIODE = 0.5       # secondes - periode du clignotement/bip

# === PID (Tier 2) - correction de direction proportionnelle a erreur_x ===
# Valeurs de depart raisonnables, A CALIBRER en test reel (couloir)
PID_KP = 0.5    # gain proportionnel
PID_KI = 0.0    # gain integral (desactive au depart)
PID_KD = 0.1    # gain derive (amortissement)
PID_OUTPUT_MAX = 100  # cm/PWM - borne max de la correction differentielle


class VisionTracker:
    def __init__(self):
        self.picam = Picamera2()
        config = self.picam.create_video_configuration(
            main={"size": (1280, 720)},
            raw={"size": (2304, 1296)}
        )
        self.picam.configure(config)
        self.picam.start()

        print("[INFO] Chargement modele ONNX...")
        self.session = ort.InferenceSession(
            'yolov8n.onnx',
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        print("[INFO] Modele ONNX OK.")

        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )

        self.prev_gray = None
        self.prev_points = None
        self.target_center = None
        self.last_yolo_frame = -100
        self.frame_count = 0
        self.yolo_interval = 10
        self.frames_sans_cible = 0
        self.max_frames_sans_cible = 5
        self.conf_threshold = 0.35

        # --- Machine a etats ---
        # Etats possibles : "PERDU" / "SUIVRE" / "TROP_PROCHE"
        self.etat = "PERDU"
        self.recul_start_time = None
        self.perdu_depuis = None  # timestamp d'entree en PERDU (pour EF_04)
        self.signal_actif = False
        self.signal_on = False
        self.signal_last_toggle = 0.0

        # --- ENF_01 : mesure latence vision -> ordre moteur ---
        self.last_latence_ms = None

        # --- PID (Tier 2) ---
        self.pid_erreur_precedente = 0.0
        self.pid_integrale = 0.0
        self.pid_last_time = time.time()

        # --- EF_04 : Buzzer + LED RGB (GPIO Pi) ---
        if GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(PIN_BUZZER, GPIO.OUT)
            GPIO.setup(PIN_LED_R, GPIO.OUT)
            GPIO.setup(PIN_LED_G, GPIO.OUT)
            GPIO.setup(PIN_LED_B, GPIO.OUT)
            self._signal_off()
            print("[INFO] GPIO buzzer/LED OK.")
        else:
            print("[AVERTISSEMENT] RPi.GPIO indisponible - signal EF_04 desactive.")

        # Distance ultrasons lue en parallele
        self.distance_cm = -1
        self.dist_lock = threading.Lock()

        self.arduino = None
        try:
            self.arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            time.sleep(2)
            print("[INFO] Arduino OK.")
            # Thread lecture distance
            self.dist_thread = threading.Thread(target=self._read_distance_loop, daemon=True)
            self.dist_thread.start()
        except:
            print("[AVERTISSEMENT] Mode simulation.")

    def _read_distance_loop(self):
        """Thread qui lit en continu les messages DIST:XX de l'Arduino."""
        while running:
            try:
                if self.arduino and self.arduino.in_waiting > 0:
                    line = self.arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith("DIST:"):
                        val = int(line.split(":")[1])
                        with self.dist_lock:
                            self.distance_cm = val
            except:
                pass
            time.sleep(0.01)

    def get_distance(self):
        with self.dist_lock:
            return self.distance_cm

    def send(self, cmd):
        if self.arduino:
            self.arduino.write(cmd)

    def _signal_off(self):
        """Coupe completement buzzer + LED."""
        if not GPIO_AVAILABLE:
            return
        GPIO.output(PIN_BUZZER, GPIO.LOW)
        GPIO.output(PIN_LED_R, GPIO.LOW)
        GPIO.output(PIN_LED_G, GPIO.LOW)
        GPIO.output(PIN_LED_B, GPIO.LOW)
        self.signal_on = False

    def _signal_toggle(self):
        """Inverse l'etat buzzer + LED rouge (clignotement)."""
        if not GPIO_AVAILABLE:
            return
        self.signal_on = not self.signal_on
        GPIO.output(PIN_BUZZER, GPIO.HIGH if self.signal_on else GPIO.LOW)
        GPIO.output(PIN_LED_R, GPIO.HIGH if self.signal_on else GPIO.LOW)
        GPIO.output(PIN_LED_G, GPIO.LOW)
        GPIO.output(PIN_LED_B, GPIO.LOW)

    def update_signal_detresse(self):
        """
        EF_04 : si en etat PERDU depuis plus de PERDU_DELAI_SIGNAL secondes,
        fait clignoter LED rouge + bip buzzer (non bloquant, toggle par frame).
        Coupe le signal immediatement si on quitte PERDU.
        """
        now = time.time()

        if self.etat != "PERDU":
            self.perdu_depuis = None
            if self.signal_actif:
                self.signal_actif = False
                self._signal_off()
            return

        if self.perdu_depuis is None:
            self.perdu_depuis = now
            return

        if (now - self.perdu_depuis) >= PERDU_DELAI_SIGNAL:
            if not self.signal_actif:
                self.signal_actif = True
                self.signal_last_toggle = now
                print("[EF_04] Signal de detresse active (PERDU > "
                      f"{PERDU_DELAI_SIGNAL}s)")
            if (now - self.signal_last_toggle) >= SIGNAL_PERIODE:
                self._signal_toggle()
                self.signal_last_toggle = now

    def stop(self):
        self.send(b"STOP\n")
        self._signal_off()

    def yolo_detect(self, frame):
        img = cv2.resize(frame, (320, 320))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis]
        outputs = self.session.run(None, {self.input_name: img})[0]
        outputs = outputs[0].T
        best_box = None
        best_conf = self.conf_threshold
        h, w = frame.shape[:2]
        for det in outputs:
            scores = det[4:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if class_id == 0 and conf > best_conf:
                best_conf = conf
                cx, cy, bw, bh = det[:4]
                x1 = int((cx - bw / 2) / 320 * w)
                y1 = int((cy - bh / 2) / 320 * h)
                x2 = int((cx + bw / 2) / 320 * w)
                y2 = int((cy + bh / 2) / 320 * h)
                best_box = (x1, y1, x2, y2)
        return best_box

    def compute_pid(self, erreur_x):
        """
        Calcule la correction PID a partir de erreur_x.
        Retourne une correction signee, bornee a +-PID_OUTPUT_MAX.
        Positive si erreur_x > 0 (cible a droite -> tourne a droite).
        """
        now = time.time()
        dt = now - self.pid_last_time
        if dt <= 0:
            dt = 1e-3  # garde-fou division par zero

        erreur = float(erreur_x)

        # Terme proportionnel
        p_term = PID_KP * erreur

        # Terme integral
        self.pid_integrale += erreur * dt
        i_term = PID_KI * self.pid_integrale

        # Terme derive
        d_term = PID_KD * (erreur - self.pid_erreur_precedente) / dt

        correction = p_term + i_term + d_term

        # Bornage
        correction = max(-PID_OUTPUT_MAX, min(PID_OUTPUT_MAX, correction))

        self.pid_erreur_precedente = erreur
        self.pid_last_time = now

        return correction

    def decide_command(self, center, centre_ecran_x, dist):
        """
        Machine a etats : SUIVRE / TROP_PROCHE / PERDU.
        Retourne (cmd, debug_str) ou (None, debug_str) si rien a envoyer.
        """
        # --- Transition vers PERDU : pas de cible ---
        if center is None:
            if self.etat != "PERDU":
                print(f"[ETAT] {self.etat} -> PERDU")
            self.etat = "PERDU"
            self.recul_start_time = None
            return b"STOP\n", "PERDU"

        erreur_x = center[0] - centre_ecran_x

        # --- Etat TROP_PROCHE : on est en train de reculer ---
        if self.etat == "TROP_PROCHE":
            # Securite anti-mur : timeout 2s
            if self.recul_start_time is not None and \
               (time.time() - self.recul_start_time) > RECUL_TIMEOUT:
                print("[ETAT] TROP_PROCHE -> SUIVRE (timeout recul 2s, securite)")
                self.etat = "SUIVRE"
                self.recul_start_time = None
                # on retombe dans la logique SUIVRE ci-dessous sur cette meme frame
            # Retour a SUIVRE si on s'est suffisamment eloigne
            elif dist > 0 and dist > DIST_RETOUR_SUIVRE:
                print(f"[ETAT] TROP_PROCHE -> SUIVRE (dist={dist}cm > {DIST_RETOUR_SUIVRE}cm)")
                self.etat = "SUIVRE"
                self.recul_start_time = None
            else:
                # Continue de reculer
                return b"RECULER\n", f"TROP_PROCHE dist={dist}cm"

        # --- Etat SUIVRE (ou venant de transitionner depuis TROP_PROCHE) ---
        if self.etat != "TROP_PROCHE":
            if self.etat != "SUIVRE":
                print(f"[ETAT] {self.etat} -> SUIVRE")
            self.etat = "SUIVRE"

            # Transition vers TROP_PROCHE (verifiee avant le calcul PID,
            # car en TROP_PROCHE on ne pilote plus via PID mais RECULER)
            if dist > 0 and dist < DIST_TROP_PRES:
                print(f"[ETAT] SUIVRE -> TROP_PROCHE (dist={dist}cm < {DIST_TROP_PRES}cm)")
                self.etat = "TROP_PROCHE"
                self.recul_start_time = time.time()
                return b"RECULER\n", f"TROP_PROCHE dist={dist}cm"

            # --- PID : correction differentielle proportionnelle a erreur_x ---
            correction = self.compute_pid(erreur_x)

            # Vitesse de base selon distance
            if dist > 0 and DIST_TROP_PRES <= dist <= DIST_AVANCER:
                vitesse_base = 0  # zone de confort : rotation sur place si besoin
            else:
                # dist > DIST_AVANCER (loin) ou dist <= 0 (inconnue) -> avance
                vitesse_base = VITESSE_AVANCE

            # erreur_x > 0 (cible a droite) -> roue droite plus rapide, gauche plus lente
            vitesse_gauche = int(vitesse_base - correction)
            vitesse_droite = int(vitesse_base + correction)

            # Bornage final securite
            vitesse_gauche = max(-PWM_MAX, min(PWM_MAX, vitesse_gauche))
            vitesse_droite = max(-PWM_MAX, min(PWM_MAX, vitesse_droite))

            cmd = f"M:{vitesse_gauche},{vitesse_droite}\n".encode()
            return cmd, (f"SUIVRE ex={erreur_x} dist={dist}cm "
                          f"PID_corr={correction:.1f} "
                          f"M=({vitesse_gauche},{vitesse_droite})")

        # Cas residuel (ne devrait pas arriver, garde-fou)
        return b"STOP\n", f"ETAT={self.etat} (fallback)"

    def process_one_frame(self):
        t_frame_start = time.time()

        frame = self.picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (320, 240))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        largeur = frame.shape[1]
        centre_ecran_x = largeur // 2

        fg_mask = self.mog2.apply(frame)
        mouvement = np.sum(fg_mask > 200) > 300

        center = None
        source = "OF"

        if mouvement and self.frame_count % self.yolo_interval == 0:
            box = self.yolo_detect(frame)
            if box is not None:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                self.target_center = (cx, cy)
                self.last_yolo_frame = self.frame_count
                self.prev_points = np.array([[cx, cy]], dtype=np.float32).reshape(-1, 1, 2)
                self.prev_gray = gray.copy()
                center = (cx, cy)
                self.frames_sans_cible = 0
                source = "YOLO"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "YOLO", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        elif self.prev_points is not None and self.prev_gray is not None:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
            )
            if new_points is not None and status[0][0] == 1:
                cx, cy = int(new_points[0][0][0]), int(new_points[0][0][1])

                # --- FIX : detection saut impossible (repositionnement manuel) ---
                old_cx, old_cy = self.prev_points[0][0]
                saut = float(np.hypot(cx - old_cx, cy - old_cy))

                if saut > SAUT_OF_MAX:
                    print(f"[OF] Saut impossible detecte ({saut:.0f}px > {SAUT_OF_MAX}px) - cible perdue forcee")
                    self.frames_sans_cible += 1
                    self.prev_points = None
                elif 0 < cx < frame.shape[1] and 0 < cy < frame.shape[0]:
                    self.target_center = (cx, cy)
                    self.prev_points = new_points
                    self.prev_gray = gray.copy()
                    center = (cx, cy)
                    self.frames_sans_cible = 0
                    cv2.circle(frame, (cx, cy), 8, (255, 165, 0), -1)
                else:
                    self.frames_sans_cible += 1
                    self.prev_points = None
            else:
                self.frames_sans_cible += 1
                self.prev_points = None
        else:
            if not mouvement:
                self.frames_sans_cible += 1

        self.frame_count += 1

        # Lecture distance ultrasons
        dist = self.get_distance()

        # Perte de cible confirmee (trop de frames sans rien)
        if center is None and self.frames_sans_cible >= self.max_frames_sans_cible:
            center = None  # force le passage en PERDU dans decide_command
        elif center is None and self.frames_sans_cible < self.max_frames_sans_cible:
            # Pas encore assez de frames sans cible : on ne redecide pas,
            # on laisse l'etat courant et on n'envoie rien (evite spam STOP)
            cmd, debug = None, f"transitoire SC={self.frames_sans_cible}"
        if center is not None or self.frames_sans_cible >= self.max_frames_sans_cible:
            cmd, debug = self.decide_command(center, centre_ecran_x, dist)

        if cmd is not None and self.frame_count % 2 == 0:
            latence_ms = (time.time() - t_frame_start) * 1000.0
            self.last_latence_ms = latence_ms
            print(f"[CMD] {cmd.decode().strip()} | {debug} | etat={self.etat} | latence={latence_ms:.1f}ms")
            self.send(cmd)

        # EF_04 : signal de detresse (buzzer + LED) si PERDU prolonge
        self.update_signal_detresse()

        # Overlay
        signal_txt = " SIGNAL!" if self.signal_actif else ""
        lat_txt = f" L:{self.last_latence_ms:.0f}ms" if self.last_latence_ms is not None else ""
        status_txt = f"F:{self.frame_count} D:{dist}cm SC:{self.frames_sans_cible} ETAT:{self.etat}{signal_txt}{lat_txt}"
        cv2.putText(frame, status_txt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()


def gen_frames():
    global tracker_instance, running
    while running:
        try:
            frame_bytes = tracker_instance.process_one_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.02)
        except Exception as e:
            print(f"[ERREUR] {e}")
            break


@app.route('/')
def index():
    return '<body style="background:#000;text-align:center;"><img src="/video_feed" style="width:80%"></body>'


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    tracker_instance = VisionTracker()
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        running = False
        if tracker_instance:
            tracker_instance.stop()
            print("[INFO] STOP envoye - arret propre")
        if GPIO_AVAILABLE:
            GPIO.cleanup()