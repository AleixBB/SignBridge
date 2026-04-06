import os
import io
import time
import logging
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Importar clasificador y las constantes que vienen del otro archivo
from clasificador import (
    cargar_letras,
    Comparador,
    DetectorEstable,
    K_NEIGHBORS
)

# CONFIGURACIÓN FASTAPI Y LOGGING 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI() # Inicialización de FastAPI
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"], # Permitir todas las fuentes (en producción, restringir a dominios específicos)
    allow_methods=["*"], # Permitir todos los métodos HTTP
    allow_headers=["*"], # Permitir todos los encabezados
)

# INICIALIZACIÓN IA
print(f"Cargando dataset (K={K_NEIGHBORS})...")
dataset = cargar_letras()
comparador = Comparador(dataset) # Usa el K de clasificador.py
detector = DetectorEstable()     # Usa los umbrales de clasificador.py

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic( # model_complexity 1 es más lento pero MUCHO más preciso que 0
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5, # MediaPipe detecta la mano con al menos este nivel de confianza
    min_tracking_confidence=0.5 # MediaPipe mantiene el seguimiento de la mano con al menos este nivel de confianza
)

# ESTABILIZACIÓN EXTRA
buffer_temporal = deque(maxlen=1) # Solo el último vector, pero podrías aumentar a 3-5 para suavizar aún más (con el costo de latencia)
frame_counter = 0

class DetectorEstabilizado:
    def __init__(self):
        self.ultima_letra = None
        self.tiempo_ultima = 0
        self.buffer_letras = deque(maxlen=2) # Buffer para las últimas letras detectadas
        self.min_tiempo = 0.7 # Tiempo mínimo entre cambios de letra para evitar oscilaciones rápidas

    def actualizar(self, letra, confianza):
        ahora = time.time()
        if not letra: return None, 0, False
        
        self.buffer_letras.append((letra, confianza))
        conteo = {}
        for l, c in self.buffer_letras:
            conteo[l] = conteo.get(l, 0) + 1
            
        letra_estable = max(conteo, key=conteo.get)
        conf_media = np.mean([c for l, c in self.buffer_letras if l == letra_estable])
        
        if letra_estable != self.ultima_letra:
            if ahora - self.tiempo_ultima > self.min_tiempo:
                self.ultima_letra = letra_estable
                self.tiempo_ultima = ahora
                return letra_estable, conf_media, True
        return letra_estable, conf_media, False

detector_estabilizado = DetectorEstabilizado()

# PROCESAMIENTO IMAGEN
def resize_letterbox(image, target_size=(320, 240)): # MediaPipe funciona bien con 320x240
    h_raw, w_raw = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w_raw, th / h_raw)
    new_w, new_h = int(w_raw * scale), int(h_raw * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x_off, y_off = (tw - new_w) // 2, (th - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# ENDPOINT
@app.post("/process_frame") # Endpoint para procesar un frame enviado desde el frontend
async def process_frame(file: UploadFile = File(...)): # Recibe un archivo de imagen (frame) enviado desde el frontend
    global frame_counter # Contador de frames para debug
    try:
        # 1. Cargar y procesar imagen
        contents = await file.read() # Leer el contenido del archivo enviado (frame)
        pil_img = Image.open(io.BytesIO(contents)) # Abrir la imagen con PIL
        frame_raw = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR) # Convertir la imagen a formato BGR para OpenCV
        frame_bgr = resize_letterbox(frame_raw) # Redimensionar la imagen para MediaPipe (320x240) manteniendo la relación de aspecto
        # 2. Inferencia MediaPipe
        results = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)) # MediaPipe necesita RGB, así que convertimos antes de procesar

        letra, confianza, letra_cambiada, has_hand = None, 0, False, False # Variables para almacenar el resultado de la clasificación y si se detectó una mano

        # 3. Clasificación
        hands = []
        if results.right_hand_landmarks: hands.append((results.right_hand_landmarks, [12, 14, 16])) # Si MediaPipe detecta la mano derecha, la añadimos a la lista de manos a procesar junto con los índices de los landmarks del brazo derecho
        if results.left_hand_landmarks: hands.append((results.left_hand_landmarks, [11, 13, 15])) # Si MediaPipe detecta la mano izquierda, la añadimos a la lista de manos a procesar junto con los índices de los landmarks del brazo izquierdo

        if hands and results.pose_landmarks:
            has_hand = True
            mejor_conf, mejor_letra, cambio = 0, None, False

            for hand_landmarks, arm_idx in hands: # Para cada mano detectada (puede haber una o dos), procesamos los landmarks de la mano y del brazo correspondiente
                brazo = [c for i in arm_idx for c in [results.pose_landmarks.landmark[i].x, 
                                                       results.pose_landmarks.landmark[i].y, 
                                                       results.pose_landmarks.landmark[i].z]]
                mano = [c for lm in hand_landmarks.landmark for c in [lm.x, lm.y, lm.z]]

                vector = np.array(brazo + mano) # Crear un vector con las coordenadas del brazo y la mano (9 del brazo + 63 de la mano = 72 dimensiones)
                buffer_temporal.append(vector) # Añadir el vector al buffer temporal para estabilización
                vector_prom = np.mean(buffer_temporal, axis=0) # Calcular el vector promedio de los últimos frames para suavizar la detección

                res_knn = comparador.comparar(vector_prom) # Obtener los resultados del clasificador KNN para el vector promedio
                if res_knn:
                    l_temp, c_temp = detector.actualizar(res_knn) # Actualizar el detector estable con los resultados del KNN
                    if l_temp:
                        l_act, c_act, c_bool = detector_estabilizado.actualizar(l_temp, c_temp) # Actualizar el detector estabilizado con la letra y confianza obtenida del detector estable
                        if c_act > mejor_conf:
                            mejor_conf, mejor_letra, cambio = c_act, l_act, c_bool

            letra, confianza, letra_cambiada = mejor_letra, mejor_conf, cambio
        else:
            buffer_temporal.clear()
            detector.buffer.clear()

        return JSONResponse(content={ # Devolver la letra detectada, su confianza, si la letra ha cambiado respecto a la última detectada y si se ha detectado una mano
            "letter": letra if letra else "-",
            "confidence": float(confianza),
            "letter_changed": letra_cambiada,
            "has_hand": has_hand,
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(content={"letter": "-", "confidence": 0, "letter_changed": False, "has_hand": False})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)