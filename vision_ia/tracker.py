import cv2
import numpy as np
import onnxruntime as ort
import serial
import time
import threading
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)
tracker_instance = None
running = True

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

        # Distance ultrasons lue en parallèle
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

    def stop(self):
        self.send(b"STOP\n")

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
                x1 = int((cx - bw/2) / 320 * w)
                y1 = int((cy - bh/2) / 320 * h)
                x2 = int((cx + bw/2) / 320 * w)
                y2 = int((cy + bh/2) / 320 * h)
                best_box = (x1, y1, x2, y2)
        return best_box

    def process_one_frame(self):
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
                cv2.putText(frame, "YOLO", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        elif self.prev_points is not None and self.prev_gray is not None:
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_points, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
            )
            if new_points is not None and status[0][0] == 1:
                cx, cy = int(new_points[0][0][0]), int(new_points[0][0][1])
                if 0 < cx < frame.shape[1] and 0 < cy < frame.shape[0]:
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

        if center is not None and self.frame_count % 2 == 0:
            erreur_x = center[0] - centre_ecran_x

            if abs(erreur_x) > 80:
                cmd = b"GAUCHE\n" if erreur_x < 0 else b"DROITE\n"
            elif dist > 0 and dist < 50:
                cmd = b"RECULER\n"
            elif dist > 0 and dist > 120:
                cmd = b"AVANCER\n"
            elif erreur_x < -40:
                cmd = b"GAUCHE\n"
            elif erreur_x > 40:
                cmd = b"DROITE\n"
            elif dist > 0 and 50 <= dist <= 80:
                cmd = b"STOP\n"
            else:
                cmd = b"AVANCER\n"

            print(f"[CMD] {cmd.decode().strip()} | ex={erreur_x} | dist={dist}cm | {source}")
            self.send(cmd)

        elif self.frames_sans_cible >= self.max_frames_sans_cible:
            self.stop()
            if self.frames_sans_cible == self.max_frames_sans_cible:
                print("[CMD] STOP | cible perdue")

        # Overlay
        status_txt = f"F:{self.frame_count} D:{dist}cm SC:{self.frames_sans_cible}"
        cv2.putText(frame, status_txt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

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
