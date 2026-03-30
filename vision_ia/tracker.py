"""
Module de Vision IA pour le projet FollowMe.
Gère la capture vidéo, l'inférence YOLOv8 et le suivi de personnes.
"""
import cv2
import numpy as np
from ultralytics import YOLO

class VisionTracker:
    def __init__(self, source="test.mp4"):
        """Initialise le tracker de vision et le modèle IA."""
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Erreur: Impossible d'ouvrir la source vidéo {self.source}")
            
        print("[INFO] Flux vidéo initialisé avec succès.")
        
        # --- INITIALISATION DE L'IA ---
        print("[INFO] Chargement du modèle YOLOv8 Nano...")
        # Le '.pt' sera téléchargé automatiquement la première fois
        self.model = YOLO('yolov8n.pt') 

    def run(self):
        """Boucle principale de capture, d'inférence et d'affichage."""
        print("[INFO] Appuyez sur 'q' pour quitter.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[AVERTISSEMENT] Fin de la vidéo. Relancez le script.")
                break
                
            # Obtenir les dimensions de la vidéo (pour trouver le centre)
            hauteur_ecran, largeur_ecran = frame.shape[:2]
            centre_ecran_x = largeur_ecran // 2
                
            # --- INFERENCE IA ---
            results = self.model(frame, classes=[0], verbose=False)
            annotated_frame = results[0].plot()
            
            # --- LOGIQUE DE SUIVI (LE CERVEAU DU ROBOT) ---
            # Si l'IA détecte au moins une personne (len > 0)
            if len(results[0].boxes) > 0:
                # On prend la première personne détectée (la plus confiante)
                box = results[0].boxes[0]
                
                # Coordonnées du rectangle : x1 (gauche), y1 (haut), x2 (droite), y2 (bas)
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                
                # Calculs mathématiques
                centre_humain_x = (x1 + x2) // 2
                largeur_humain = x2 - x1
                
                # Calcul de l'erreur (différence entre le centre de l'écran et la personne)
                erreur_x = centre_humain_x - centre_ecran_x
                
                # --- GÉNÉRATION DES COMMANDES MOTEURS ---
                marge_morte = 50 # Tolérance de 50 pixels pour éviter que le robot tremble
                
                if largeur_humain > 250: # La personne prend beaucoup de place = elle est trop près !
                    action = "🛑 STOP (Trop près)"
                elif erreur_x > marge_morte:
                    action = "➡️ TOURNER DROITE"
                elif erreur_x < -marge_morte:
                    action = "⬅️ TOURNER GAUCHE"
                else:
                    action = "⬆️ AVANCER (Personne centrée)"
                    
                # On affiche la décision dans le terminal
                print(f"[COMMANDE] {action} | Erreur X: {erreur_x}px | Taille: {largeur_humain}px")
                
                # Bonus : On dessine une ligne cible sur la vidéo pour le debug
                cv2.line(annotated_frame, (centre_ecran_x, 0), (centre_ecran_x, hauteur_ecran), (0, 255, 0), 2)
                cv2.circle(annotated_frame, (centre_humain_x, (y1+y2)//2), 5, (0, 0, 255), -1)

            else:
                print("[COMMANDE] ❓ CIBLE PERDUE - RECHERCHE...")
            
            # Affichage
            cv2.imshow("FollowMe - Vision Edge AI", annotated_frame)
            # Condition de sortie (Touche 'q') ET frein moteur !
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
                
    def cleanup(self):
        """Libère les ressources matérielles."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Ressources libérées. Arrêt du système.")

if __name__ == "__main__":
    tracker = VisionTracker(source="test.mp4") 
    tracker.run()