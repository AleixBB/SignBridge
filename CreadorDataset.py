import cv2
import mediapipe as mp
import numpy as np
import os
import time
import csv

DATASET_FILE = "dataset_lse.csv"
OUTPUT_FOLDER = "dataset_fotos"
DURATION_SECONDS = 15
COUNTDOWN_SECONDS = 3

# Inicializacion de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_arm_hand_data(results):
    """
    Extrae 72 valores del brazo y la mano:
    - 3 puntos del brazo: hombro, codo, muñeca (3*3 = 9 valores)
    - 21 puntos de la mano (21*3 = 63 valores)
    Total = 72 valores
    """

    # Vector vacío si no se detecta nada
    blank = np.zeros(72, dtype=np.float32)

    hand_lms = None
    arm_idx = []
    flip_x = False

    # Prioridad a la mano derecha
    if results.right_hand_landmarks:
        hand_lms = results.right_hand_landmarks
        arm_idx = [12, 14, 16]

    # Si no, usamos la mano izquierda
    elif results.left_hand_landmarks:
        hand_lms = results.left_hand_landmarks
        arm_idx = [11, 13, 15]
        flip_x = True

    # Si hay mano y pose detectadas
    if hand_lms and results.pose_landmarks:

        res_arm = []

        # Extraemos hombro, codo y muñeca
        for i in arm_idx:
            lm = results.pose_landmarks.landmark[i]

            x = (1 - lm.x) if flip_x else lm.x

            res_arm.extend([x, lm.y, lm.z])

        res_hand = []

        # Extraemos los 21 puntos de la mano
        for lm in hand_lms.landmark:

            x = (1 - lm.x) if flip_x else lm.x

            res_hand.extend([x, lm.y, lm.z])

        # Devolvemos vector final
        return np.array(res_arm + res_hand, dtype=np.float32), True

    # Si no se detecta mano
    return blank, False

def ensure_csv():
    """
    Comprueba si el dataset CSV existe.
    Si no existe, crea el archivo con la cabecera.
    """

    if not os.path.exists(DATASET_FILE):

        with open(DATASET_FILE, "w", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            header = ["label"] + [f"v{i}" for i in range(72)]

            writer.writerow(header)

def init_camera():
    """
    Inicializa la cámara.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        return None

    return cap

def prepare_label_folder(label):
    """
    Crea la carpeta para guardar imágenes del gesto.
    También calcula cuántas imágenes existen ya.
    """

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    label_folder = os.path.join(OUTPUT_FOLDER, label)

    os.makedirs(label_folder, exist_ok=True)

    existing = [x for x in os.listdir(label_folder) if x.endswith(".jpg")]

    return label_folder, len(existing)

def countdown(cap, label):
    """
    Muestra una cuenta atrás antes de empezar la grabación.
    """

    start_wait = time.time()

    while time.time() - start_wait < COUNTDOWN_SECONDS:

        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        remaining = COUNTDOWN_SECONDS - int(time.time() - start_wait)

        cv2.putText(frame, f"PREPARATE: {remaining}", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        cv2.putText(frame, f"GESTO: {label}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Captura Brazo + Mano", frame)

        # ESC cancela
        if cv2.waitKey(1) & 0xFF == 27:
            return False

    return True

def draw_arm(frame, results, w, h):
    """
    Dibuja los puntos del brazo (hombro, codo y muñeca)
    """

    idx_arm = [12, 14, 16] if results.right_hand_landmarks else [11, 13, 15]

    p_pts = []

    for i in idx_arm:

        lm = results.pose_landmarks.landmark[i]

        p_pts.append((int(lm.x * w), int(lm.y * h)))

    # Dibujar articulaciones
    cv2.circle(frame, p_pts[0], 12, (0, 255, 255), -1)
    cv2.circle(frame, p_pts[1], 12, (0, 255, 0), -1)
    cv2.circle(frame, p_pts[2], 12, (255, 0, 0), -1)

    # Dibujar líneas del brazo
    cv2.line(frame, p_pts[0], p_pts[1], (255, 255, 255), 4)
    cv2.line(frame, p_pts[1], p_pts[2], (255, 255, 255), 4)

def draw_hand(frame, results):
    """
    Dibuja los 21 landmarks de la mano.
    """

    hand_lms = results.right_hand_landmarks if results.right_hand_landmarks else results.left_hand_landmarks

    mp_drawing.draw_landmarks(
        frame,
        hand_lms,
        mp_holistic.HAND_CONNECTIONS
    )

def draw_ui(frame, label, captured, time_left):
    """
    Dibuja la interfaz con información de captura.
    """

    cv2.rectangle(frame, (0, 0), (420, 90), (0, 0, 0), -1)

    cv2.putText(frame, f"Gesto: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Tiempo: {time_left}s", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Capturado: {captured}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
def capture_dataset(cap, label, label_folder, img_counter):
    """
    Función principal que captura las muestras del gesto.
    Guarda:
    - imágenes
    - vectores en el CSV
    """

    captured = 0

    with mp_holistic.Holistic(
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
    ) as holistic:

        start_time = time.time()

        with open(DATASET_FILE, "a", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            while True:

                elapsed = time.time() - start_time

                if elapsed > DURATION_SECONDS:
                    break

                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.flip(frame, 1)

                h, w, _ = frame.shape

                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                data_vector, detected = extract_arm_hand_data(results)

                if detected:

                    img_counter += 1
                    captured += 1

                    draw_arm(frame, results, w, h)

                    draw_hand(frame, results)

                    img_name = os.path.join(label_folder, f"{img_counter}.jpg")

                    cv2.imwrite(img_name, frame)

                    row = [label] + list(data_vector)

                    writer.writerow(row)

                else:

                    cv2.putText(frame,
                                "BRAZO NO DETECTADO - ALEJATE",
                                (40, h - 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2)

                time_left = int(DURATION_SECONDS - elapsed)

                draw_ui(frame, label, captured, time_left)

                cv2.imshow("Captura Brazo + Mano", frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

    return captured

def main():
    """
    Función principal del programa.
    """

    label = input("Introduce la letra o gesto: ").strip()

    if not label:
        print("Etiqueta no válida")
        return

    ensure_csv()

    label_folder, img_counter = prepare_label_folder(label)

    cap = init_camera()

    if cap is None:
        return

    if not countdown(cap, label):
        cap.release()
        cv2.destroyAllWindows()
        return

    print("Grabando...")

    captured = capture_dataset(cap, label, label_folder, img_counter)

    cap.release()
    cv2.destroyAllWindows()

    print("Proceso finalizado")
    print(f"Muestras añadidas para '{label}': {captured}")
    print(f"Dataset guardado en '{DATASET_FILE}'")
    print(f"Imágenes guardadas en '{label_folder}'")


if __name__ == "__main__":
    main()