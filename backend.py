# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import numpy as np
import io
from PIL import Image
import logging
import time
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# IMPORTAR CLASIFICADOR
# ==========================
from clasificador import (
    cargar_letras,
    Comparador,
    DetectorEstable,
    CONFIDENCE_THRESHOLD,
    STABILITY_FRAMES,
    COOLDOWN_FRAMES,
    K_NEIGHBORS,
)

print("Cargando dataset...")
dataset = cargar_letras()
print(f"Dataset cargado. Letras disponibles: {list(dataset.keys())}")

# 🔥 Reducimos K para velocidad
comparador = Comparador(dataset, k=5)

# 🔥 Más estabilidad
detector = DetectorEstable(CONFIDENCE_THRESHOLD, 3, COOLDOWN_FRAMES)

# ==========================
# MEDIAPIPE GLOBAL (CLAVE)
# ==========================
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,  # 🔥 MÁS RÁPIDO
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# BUFFERS
# ==========================
TEMPORAL_SMOOTHING = 1
buffer_temporal = deque(maxlen=TEMPORAL_SMOOTHING)

# Cache rápida
ultima_respuesta = {
    "letter": "-",
    "confidence": 0.0,
    "letter_changed": False,
    "has_hand": False,
}
ultimo_tiempo = 0

# ==========================
# DETECTOR EXTRA ESTABLE
# ==========================
class DetectorEstabilizado:
    def __init__(self):
        self.ultima_letra = None
        self.tiempo_ultima = 0
        self.buffer_letras = deque(maxlen=3)
        self.min_tiempo = 0.2

    def actualizar(self, letra, confianza):
        ahora = time.time()

        if not letra:
            return None, 0, False

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

# ==========================
# ENDPOINT PRINCIPAL
# ==========================
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    global ultima_respuesta, ultimo_tiempo

    try:
        # ⚡ CACHE (evita procesar demasiado rápido)
        if time.time() - ultimo_tiempo < 0.1:
            return JSONResponse(content=ultima_respuesta)

        contents = await file.read()
        np_img = np.array(Image.open(io.BytesIO(contents)))

        # 🔥 REDUCCIÓN DE RESOLUCIÓN (MUY IMPORTANTE)
        frame_bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (320, 240))

        results = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        letra = None
        confianza = 0
        letra_cambiada = False
        has_hand = False

        hand = results.right_hand_landmarks or results.left_hand_landmarks

        if hand and results.pose_landmarks:
            has_hand = True

            # Determinar brazo
            arm_idx = [12, 14, 16] if results.right_hand_landmarks else [11, 13, 15]

            # Extraer vector
            brazo = []
            for i in arm_idx:
                lm = results.pose_landmarks.landmark[i]
                brazo.extend([lm.x, lm.y, lm.z])

            mano = []
            for lm in hand.landmark:
                mano.extend([lm.x, lm.y, lm.z])

            vector = np.array(brazo + mano)

            # 🔥 SUAVIZADO
            buffer_temporal.append(vector)
            vector = np.mean(buffer_temporal, axis=0)

            # 🔥 CLASIFICACIÓN
            resultados = comparador.comparar(vector)

            if resultados:
                letra_temp, conf_temp = detector.actualizar(resultados)

                if letra_temp:
                    letra, confianza, letra_cambiada = detector_estabilizado.actualizar(
                        letra_temp, conf_temp
                    )

                    if letra_cambiada:
                        logger.info(f"Letra: {letra} ({confianza:.1f}%)")

        else:
            buffer_temporal.clear()
            detector.buffer.clear()

        respuesta = {
            "letter": letra if letra else "-",
            "confidence": float(confianza),
            "letter_changed": letra_cambiada,
            "has_hand": has_hand,
        }

        # Guardar cache
        ultima_respuesta = respuesta
        ultimo_tiempo = time.time()

        return JSONResponse(content=respuesta)

    except Exception as e:
        logger.error(f"Error: {e}")
        return JSONResponse(content={
            "letter": "-",
            "confidence": 0.0,
            "letter_changed": False,
            "has_hand": False,
        })


# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)