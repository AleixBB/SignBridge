import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIGURACIÓN ---
OUTPUT_FOLDER = "dataset_fotos"
VECTOR_FILE = "vectores.txt"
DURATION_SECONDS = 15  
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_arm_hand_data(results):
    """Extrae 72 valores: Hombro(0-2), Codo(3-5), Muñeca(6-8) y Mano(9-71)"""
    blank = np.zeros(72)
    hand_lms = None
    arm_idx = [] 
    flip_x = False

    # 1. Prioridad Mano Derecha, si no la Izquierda
    if results.right_hand_landmarks:
        hand_lms = results.right_hand_landmarks
        arm_idx = [12, 14, 16] # Hombro, Codo, Muñeca DERECHOS
    elif results.left_hand_landmarks:
        hand_lms = results.left_hand_landmarks
        arm_idx = [11, 13, 15] # Hombro, Codo, Muñeca IZQUIERDOS
        flip_x = True
    
    # 2. Solo guardamos si detectamos la MANO y la POSE del brazo
    if hand_lms and results.pose_landmarks:
        res_arm = []
        for i in arm_idx:
            lm = results.pose_landmarks.landmark[i]
            x = (1 - lm.x) if flip_x else lm.x
            res_arm.append([x, lm.y, lm.z])
        
        res_hand = []
        for lm in hand_lms.landmark:
            x = (1 - lm.x) if flip_x else lm.x
            res_hand.append([x, lm.y, lm.z])
            
        return np.concatenate([np.array(res_arm).flatten(), np.array(res_hand).flatten()]), True
    
    return blank, False

cap = cv2.VideoCapture(0)
img_counter = 0

# model_complexity=1 o 2 para mejor detección del brazo completo
with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # CUENTA ATRÁS
    start_wait = time.time()
    while time.time() - start_wait < 3:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"ALEJATE: {3 - int(time.time() - start_wait)}s", (150, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow("Captura Brazo + Mano", frame)
        cv2.waitKey(1)

    print("Grabando...")
    start_time = time.time()
    
    with open(VECTOR_FILE, "w") as f:
        while True:
            elapsed = time.time() - start_time
            if elapsed > DURATION_SECONDS: break

            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            data_vector, detected = extract_arm_hand_data(results)

            if detected:
                img_counter += 1

                # --- DIBUJO DEL BRAZO COMPLETO ---
                idx_arm = [12, 14, 16] if results.right_hand_landmarks else [11, 13, 15]
                p_pts = []

                for i in idx_arm:
                    lm = results.pose_landmarks.landmark[i]
                    p_pts.append((int(lm.x * w), int(lm.y * h)))

                # Dibujar articulaciones
                cv2.circle(frame, p_pts[0], 12, (0, 255, 255), -1)  # Hombro
                cv2.circle(frame, p_pts[1], 12, (0, 255, 0), -1)    # Codo
                cv2.circle(frame, p_pts[2], 12, (255, 0, 0), -1)    # Muñeca

                # Dibujar huesos
                cv2.line(frame, p_pts[0], p_pts[1], (255, 255, 255), 4)
                cv2.line(frame, p_pts[1], p_pts[2], (255, 255, 255), 4)

                # Dibujar mano
                hand_lms = results.right_hand_landmarks if results.right_hand_landmarks else results.left_hand_landmarks

                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_holistic.HAND_CONNECTIONS
                )

                # --- AHORA GUARDAR IMAGEN CON LANDMARKS ---
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{img_counter}.jpg"), frame)

                # --- GUARDAR VECTOR ---
                vector_str = ",".join([f"{v:.4f}" for v in data_vector])
                f.write(f"{img_counter}:[{vector_str}]\n")
            else:
                cv2.putText(frame, "BRAZO NO DETECTADO - ALEJATE MAS", (50, h-30), 0, 0.7, (0,0,255), 2)

            # UI
            cv2.rectangle(frame, (0,0), (320, 80), (0,0,0), -1)
            cv2.putText(frame, f"Tiempo: {int(DURATION_SECONDS-elapsed)}s", (10, 30), 0, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"Capturado: {img_counter}", (10, 65), 0, 0.8, (0,255,0), 2)

            cv2.imshow("Captura Brazo + Mano", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
print(f"Finalizado. {img_counter} muestras listas.")
