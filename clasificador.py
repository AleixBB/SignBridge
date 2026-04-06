import numpy as np
from collections import deque, Counter
import time

# ==========================================
# CONFIGURACIÓN CENTRALIZADA (Control total)
# ==========================================
DATASET_FILE = "dataset_lse.csv"
K_NEIGHBORS = 9             # Número de vecinos para KNN
CONFIDENCE_THRESHOLD = 80   # % mínimo para considerar una letra válida
STABILITY_FRAMES = 6    # Cuántos frames seguidos debe repetirse la letra
COOLDOWN_FRAMES = 0         # Frames de espera tras detectar una letra
UNKNOWN_THRESHOLD = 80      # Umbral de seguridad KNN
MARGIN_THRESHOLD = 80       # Diferencia mínima entre el 1er y 2do candidato

# Pesos para priorizar partes del cuerpo (Brazo y Mano)
FEATURE_WEIGHTS = np.array(
    [6]*9 + [6]*5 + [4]*5 + [2]*5 + [3]*3 + [1]*3 
)

# ==========================================
# EXTRACCIÓN DE FEATURES
# ==========================================
def angulo(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1, 1))

def angulo_vectores(v1, v2):
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(cosang, -1, 1))

def extraer_features(vector):
    vector = np.array(vector)
    brazo = vector[:9].reshape(3, 3)
    mano = vector[9:].reshape(21, 3)

    puntos = mano.copy()
    puntos -= puntos[0]
    escala = np.linalg.norm(puntos[9])
    if escala != 0: puntos /= escala

    # 1. Flexiones
    dedos = [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]
    flexiones = []
    for mcp, pip, dip, tip in dedos:
        flexiones.append(angulo(puntos[mcp], puntos[pip], puntos[dip]))
        flexiones.append(angulo(puntos[pip], puntos[dip], puntos[tip]))
    flexiones.append(angulo(puntos[2], puntos[3], puntos[4]))

    # 2. Distancias y Extensiones
    distancias = [
        np.linalg.norm(puntos[8]-puntos[12]), np.linalg.norm(puntos[12]-puntos[16]),
        np.linalg.norm(puntos[16]-puntos[20]), np.linalg.norm(puntos[4]-puntos[8]),
        np.linalg.norm(puntos[4]-puntos[12])
    ]
    extensiones = [np.linalg.norm(puntos[i]) for i in [8, 12, 16, 20, 4]]

    # 3. Apertura (Ángulos entre dedos)
    angulos_dedos = [
        angulo_vectores(puntos[8], puntos[12]), angulo_vectores(puntos[12], puntos[16]),
        angulo_vectores(puntos[16], puntos[20]), angulo_vectores(puntos[8], puntos[16]),
        angulo_vectores(puntos[8], puntos[20])
    ]

    # 4. Orientación Palma y Brazo
    v1, v2 = puntos[5]-puntos[0], puntos[17]-puntos[0]
    normal = np.cross(v1, v2)
    orientacion = normal / (np.linalg.norm(normal) + 1e-6)
    
    sh, el, wr = brazo
    brazo_f = [np.linalg.norm(sh-el), np.linalg.norm(el-wr), angulo(sh, el, wr)]

    return np.round(np.concatenate([flexiones, distancias, extensiones, angulos_dedos, orientacion, brazo_f]), 4)

# CLASES DE PROCESAMIENTO
def cargar_letras():
    letras = {}
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8-sig') as f:
            for linea in f:
                valores = linea.strip().split(",")
                if len(valores) != 73: continue
                letra = valores[0].strip().lower()
                features = extraer_features(np.array(valores[1:], dtype=float))
                letras.setdefault(letra, []).append(features)
    except FileNotFoundError:
        print(f"Error: No se encontró {DATASET_FILE}")
    return letras

class Comparador:
    def __init__(self, dataset, k=K_NEIGHBORS):
        self.k = k
        self.features, self.labels = [], []
        for letra, muestras in dataset.items():
            for f in muestras:
                self.features.append(f)
                self.labels.append(letra)
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        self.mean = np.mean(self.features, axis=0)
        self.std = np.std(self.features, axis=0) + 1e-6
        self.features = (self.features - self.mean) / self.std

    def comparar(self, vector):
        f_norm = (extraer_features(vector) - self.mean) / self.std
        dists = np.linalg.norm((self.features - f_norm) * FEATURE_WEIGHTS, axis=1)
        idx = np.argsort(dists)[:self.k]
        
        votos = {}
        for letra, d in zip(self.labels[idx], dists[idx]):
            votos[letra] = votos.get(letra, 0) + 1/(d + 1e-6)
        
        total = sum(votos.values())
        res = sorted({l: (v/total)*100 for l, v in votos.items()}.items(), key=lambda x: x[1], reverse=True)

        if not res or res[0][1] < UNKNOWN_THRESHOLD: return {}
        if len(res) > 1 and (res[0][1] - res[1][1]) < MARGIN_THRESHOLD: return {}
        
        return {res[0][0]: res[0][1]}

class DetectorEstable:
    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD, 
                 stability_frames=STABILITY_FRAMES, 
                 cooldown_frames=COOLDOWN_FRAMES):
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.cooldown_frames = cooldown_frames
        self.buffer = deque(maxlen=stability_frames)
        self.cooldown = 0
        self.ultima = None

    def actualizar(self, resultados): # resultados es un diccionario {letra: confianza} del comparador
        if not resultados: # Si no hay resultados válidos, reseteamos el buffer y devolvemos None
            self.buffer.clear() # Reseteamos el buffer para evitar que letras anteriores influyan en la próxima detección
            return None, 0 # Devolvemos None y confianza 0 para indicar que no se ha detectado una letra válida
            
        letra = max(resultados, key=resultados.get) # Obtenemos la letra con mayor confianza del comparador
        conf = resultados[letra] # Obtenemos la confianza de esa letra

        if self.cooldown > 0:  # Si estamos en periodo de cooldown, no actualizamos la letra detectada y simplemente decrementamos el cooldown
            self.cooldown -= 1 # Decrementamos el cooldown para contar los frames de espera
            return self.ultima, conf # Devolvemos la última letra establecida y la confianza actual, sin actualizar la letra detectada

        if conf < self.confidence_threshold: # Si la confianza es menor que el umbral, reseteamos el buffer y no actualizamos la letra detectada
            self.buffer.clear() # Reseteamos el buffer para evitar que letras anteriores influyan en la próxima detección
            return None, conf # Devolvemos None para indicar que no se ha detectado una letra válida, pero mantenemos la confianza para información adicional

        self.buffer.append(letra)
        if len(self.buffer) < self.stability_frames:
            return self.ultima, conf

        letra_estable = Counter(self.buffer).most_common(1)[0][0] # Obtenemos la letra más común en el buffer, que es la letra establecida
        if letra_estable != self.ultima:
            self.ultima = letra_estable
            self.cooldown = self.cooldown_frames
            
        return self.ultima, conf