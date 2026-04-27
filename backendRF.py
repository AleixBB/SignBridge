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

from clasificadorRF import ComparadorML

# CONFIGURACIÓN
TARGET_SIZE = (320, 240) # Tamaño al que se redimensionan las imagenes para MediaPipe 
UMBRAL_CONFIANZA = 35
FRAMES_NECESARIOS = 3  # 3 frames iguales con la misma letra para enviar a frontend

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI() # Inicialización de FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# INICIALIZACIÓN
# ==========================================
print("Cargando modelo...")
comparador = ComparadorML() # Carga el modelo de clasificador entrenado (usa el mismo que en clasificador.py)

mp_holistic = mp.solutions.holistic #inicializacion de mp holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0, # 0 es más rápido pero menos preciso, 1 es más lento pero mucho más preciso
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    enable_segmentation=False,
    refine_face_landmarks=False,
)

# ==========================================
# BUFFER SIMPLE: solo guarda las últimas letras
# ==========================================
buffer_letras = deque(maxlen=FRAMES_NECESARIOS) #guarda ultimas letras detactadas sin importar confianza
ultima_letra_enviada = None

def resize_letterbox(image): # Redimensiona la imagen manteniendo la relación de aspecto y añadiendo padding
    h_raw, w_raw = image.shape[:2]
    tw, th = TARGET_SIZE
    scale = min(tw / w_raw, th / h_raw)
    new_w, new_h = int(w_raw * scale), int(h_raw * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x_off, y_off = (tw - new_w) // 2, (th - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

@app.post("/process_frame") # Endpoint para procesar cada frame enviado desde el frontend
async def process_frame(file: UploadFile = File(...)):
    global buffer_letras, ultima_letra_enviada
    
    try:
        # Leer imagen
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        frame_raw = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        frame_bgr = resize_letterbox(frame_raw) 
        
        # Procesar con MediaPipe
        results = holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        letra_actual = None
        confianza_actual = 0
        has_hand = False
        
        # Detectar manos
        hands = []
        if results.right_hand_landmarks:
            hands.append((results.right_hand_landmarks, [12, 14, 16])) #indices de hombro, codo, muñeca para brazo derecho
        if results.left_hand_landmarks:
            hands.append((results.left_hand_landmarks, [11, 13, 15])) #indices de hombro, codo, muñeca para brazo izquierdo
        
        # Procesar si hay mano
        if hands and results.pose_landmarks:
            has_hand = True
            mejor_conf, mejor_letra = 0, None
            
            for hand_landmarks, arm_idx in hands:
                # Extraer brazo (9 coordenadas)
                brazo = []
                for i in arm_idx:
                    brazo.extend([
                        results.pose_landmarks.landmark[i].x,
                        results.pose_landmarks.landmark[i].y,
                        results.pose_landmarks.landmark[i].z
                    ])
                
                # Extraer mano (63 coordenadas)
                mano = []
                for lm in hand_landmarks.landmark:
                    mano.extend([lm.x, lm.y, lm.z])
                
                vector = np.array(brazo + mano)
                
                # Clasificar
                res = comparador.comparar(vector)
                
                if res:
                    letra = list(res.keys())[0]
                    conf = list(res.values())[0]
                    if conf > mejor_conf:
                        mejor_conf, mejor_letra = conf, letra
            
            letra_actual = mejor_letra
            confianza_actual = mejor_conf
        
        # SIMPLE: Verificar 3 frames iguales
        letra_final = None
        confianza_final = 0
        debe_enviar = False
        
        if letra_actual and confianza_actual >= UMBRAL_CONFIANZA:
            buffer_letras.append(letra_actual)
            
            # Mostrar progreso
            print(f"🔍 {letra_actual.upper()} ({confianza_actual:.1f}%) [buffer: {len(buffer_letras)}/{FRAMES_NECESARIOS}]")
            
            # Verificar si tenemos 3 frames iguales
            if len(buffer_letras) == FRAMES_NECESARIOS:
                letras_lista = list(buffer_letras)
                if letras_lista[0] == letras_lista[1] == letras_lista[2]:
                    letra_final = letras_lista[0]
                    confianza_final = confianza_actual
                    
                    # No enviar la misma letra repetida
                    if letra_final != ultima_letra_enviada:
                        debe_enviar = True
                        ultima_letra_enviada = letra_final
                        print(f"✅ ENVIANDO: {letra_final.upper()} ({confianza_final:.1f}%)")
                    else:
                        print(f"   ⏭️ {letra_final.upper()} ya enviada")
        else:
            # Si no hay letra válida, reiniciar buffer
            buffer_letras.clear()
        
        return JSONResponse(content={
            "letter": letra_final if letra_final else "-",
            "confidence": float(confianza_final),
            "letter_changed": debe_enviar,
            "has_hand": has_hand
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return JSONResponse(content={
            "letter": "-", "confidence": 0, 
            "letter_changed": False, "has_hand": False
        })

if __name__ == "__main__":
    print(f"   {FRAMES_NECESARIOS} frames iguales para enviar")
    print(f"   Confianza mínima: {UMBRAL_CONFIANZA}%")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")