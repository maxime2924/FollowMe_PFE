import cv2
import numpy as np
from ultralytics import YOLO
import serial
import time
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)
tracker_instance = None

class VisionTracker:
    def __init__(self, source="test.mp4"):
        self.source = source
        self.picam = Picamera2()
        self.picam.configure(self.picam.create_video_configuration(main={"size": (320, 240)}))
        self.picam.start()
        
        # Chargement du modèle NCNN (ton modèle ultra-rapide)
        print("[INFO] Chargement du modèle YOLOv8...")
        self.model = YOLO('yolov8n.pt')

        # Initialisation du Tracker OpenCV (Léger)
        self.tracker = cv2.TrackerMIL_create()
        self.tracking_active = False
        self.frame_count = 0
        self.detection_interval = 15 # On relance YOLO toutes les 15 images

        self.arduino = None
        try:
            self.arduino = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            time.sleep(2)
            print("[INFO] Arduino OK.")
        except:
            print("[AVERTISSEMENT] Mode simulation.")

    def process_one_frame(self):
        frame = self.picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Redimensionnement pour Flask
        frame = cv2.resize(frame, (320, 240))
        hauteur_ecran, largeur_ecran = frame.shape[:2]
        centre_ecran_x = largeur_ecran // 2

        success = False
        box_to_draw = None

        # LOGIQUE HYBRIDE
        # Si on a perdu la cible ou si c'est le moment de rafraîchir avec YOLO
        if not self.tracking_active or self.frame_count % self.detection_interval == 0:
            results = self.model(frame, classes=[0], verbose=False, imgsz=320)

            if len(results[0].boxes) > 0:
                # On récupère la boîte YOLO
                box = results[0].boxes[0].xyxy[0].int().tolist()
                # On convertit pour le Tracker OpenCV [x, y, w, h]
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y

                # On (re)démarre le tracker léger
                if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0] and w > 10 and h > 10:
                    self.tracker = cv2.TrackerMIL_create()
                    self.tracker.init(frame, (x, y, w, h))
                self.tracking_active = True
                box_to_draw = (x, y, w, h)
                success = True
        else:
            # On utilise le suivi ultra-rapide (pas d'IA ici !)
            try:
                success, bbox = self.tracker.update(frame)
                if success:
                    box_to_draw = [int(v) for v in bbox]
                else:
                    self.tracking_active = False
            except cv2.error:
                self.tracking_active = False
                success = False

        self.frame_count += 1

        # DESSIN ET COMMANDE ARDUINO
        if success and box_to_draw:
            x, y, w, h = box_to_draw
            centre_humain_x = x + (w // 2)
            erreur_x = centre_humain_x - centre_ecran_x

            # Dessin
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(frame, (centre_humain_x, y + h//2), 5, (0, 0, 255), -1)

            # Envoi Arduino (Optionnel : limiter la fréquence des messages)
            if self.arduino and self.frame_count % 3 == 0:
                if w > 150: cmd = b"STOP\n"
                elif erreur_x > 40: cmd = b"DROITE\n"
                elif erreur_x < -40: cmd = b"GAUCHE\n"
                else: cmd = b"AVANCER\n"
                self.arduino.write(cmd)

        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

def gen_frames():
    global tracker_instance
    while True:
        start_time = time.time()
        frame_bytes = tracker_instance.process_one_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # LIMITATION À 20 FPS (0.05s par image)
        # Cela laisse 50% de temps au processeur pour refroidir entre chaque image
        elapsed = time.time() - start_time
        time.sleep(max(0, 0.05 - elapsed))

@app.route('/')
def index():
    return '<body style="background:#000;text-align:center;"><img src="/video_feed" style="width:80%"></body>'

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    tracker_instance = VisionTracker("camera")
    app.run(host='0.0.0.0', port=5000, threaded=True)