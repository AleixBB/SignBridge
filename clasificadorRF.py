import numpy as np
import pickle
import os

# CONFIGURACIÓN
MODELO_PATH = "modelo_lse_rf.pkl"  #modelo que usaremos basado en RandomForest
CONFIDENCE_THRESHOLD = 35  # Confianza mínima para aceptar una letra

# EXTRACCIÓN DE FEATURES
def angulo(a, b, c): # Calcula el ángulo entre tres puntos (en radianes)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1, 1))

def angulo_vectores(v1, v2): # Calcula el ángulo entre dos vectores (en radianes)
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.arccos(np.clip(cosang, -1, 1))

def extraer_features(vector): #extrae características relevantes del vector crudo (brazo + mano) para la clasificación
    vector = np.array(vector, dtype=float)
    #proceso normalizacion
    if len(vector) >= 12:
        brazo_raw = vector[:9].reshape(3, 3)
        mano_raw = vector[9:].reshape(21, 3)
        muneca = mano_raw[0]
        mano_normalizada = mano_raw - muneca
        distancia_referencia = np.linalg.norm(mano_normalizada[12])
        if distancia_referencia > 0.01:
            mano_normalizada = mano_normalizada / distancia_referencia
        vector_normalizado = np.concatenate([brazo_raw.flatten(), mano_normalizada.flatten()])
    else:
        vector_normalizado = vector
    
    brazo = vector_normalizado[:9].reshape(3, 3)
    mano = vector_normalizado[9:].reshape(21, 3)
    puntos = mano.copy()
    puntos -= puntos[0]
    escala = np.linalg.norm(puntos[9])
    if escala != 0:
        puntos /= escala

    dedos = [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]
    flexiones = []
    for mcp, pip, dip, tip in dedos:
        flexiones.append(angulo(puntos[mcp], puntos[pip], puntos[dip]))
        flexiones.append(angulo(puntos[pip], puntos[dip], puntos[tip]))
    flexiones.append(angulo(puntos[2], puntos[3], puntos[4]))

    distancias = [
        np.linalg.norm(puntos[8]-puntos[12]), np.linalg.norm(puntos[12]-puntos[16]),
        np.linalg.norm(puntos[16]-puntos[20]), np.linalg.norm(puntos[4]-puntos[8]),
        np.linalg.norm(puntos[4]-puntos[12])
    ]
    
    extensiones = [np.linalg.norm(puntos[i]) for i in [8, 12, 16, 20, 4]]

    angulos_dedos = [
        angulo_vectores(puntos[8], puntos[12]), angulo_vectores(puntos[12], puntos[16]),
        angulo_vectores(puntos[16], puntos[20]), angulo_vectores(puntos[8], puntos[16]),
        angulo_vectores(puntos[8], puntos[20])
    ]

    v1, v2 = puntos[5]-puntos[0], puntos[17]-puntos[0]
    normal = np.cross(v1, v2)
    norm_norm = np.linalg.norm(normal)
    if norm_norm > 1e-6:
        orientacion = normal / norm_norm
    else:
        orientacion = np.zeros(3)
    
    sh, el, wr = brazo
    brazo_f = [np.linalg.norm(sh-el), np.linalg.norm(el-wr), angulo(sh, el, wr)]

    features = np.concatenate([flexiones, distancias, extensiones, angulos_dedos, orientacion, brazo_f])
    features = np.nan_to_num(features, 0)
    
    return features

# CLASIFICADOR 
# Carga el modelo de clasificación y lo utiliza para comparar un vector crudo con las letras conocidas, devolviendo la letra con mayor confianza si supera el umbral establecido.
class ComparadorML:
    def __init__(self, modelo_path=MODELO_PATH):
        self.modelo = None
        self.scaler = None
        self.classes = None
        
        if os.path.exists(modelo_path):
            self.cargar_modelo(modelo_path)
        else:
            raise FileNotFoundError(f"No se encontró el modelo en {modelo_path}")
    
    def cargar_modelo(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.modelo = data['modelo']
            self.scaler = data['scaler']
            self.classes = data['classes']
        print(f"✓ Modelo cargado con {len(self.classes)} clases")
    
    def comparar(self, vector_crudo):
        # Compara un vector crudo con las letras conocidas
        try:
            features_raw = extraer_features(vector_crudo) # Extrae características relevantes del vector crudo
            if features_raw.ndim == 1: 
                features_raw = features_raw.reshape(1, -1) # Asegura que el vector tenga la forma correcta para el modelo
            
            features_scaled = self.scaler.transform(features_raw) # Escala las características utilizando el scaler cargado
            probabilidades = self.modelo.predict_proba(features_scaled)[0] # Obtiene las probabilidades de cada clase para el vector dado
            
            mejor_idx = np.argmax(probabilidades) # Índice de la clase con mayor probabilidad
            mejor_letra = self.classes[mejor_idx]
            mejor_prob = probabilidades[mejor_idx] * 100
            
            if mejor_prob < CONFIDENCE_THRESHOLD:
                return {}
            
            return {mejor_letra: mejor_prob} # Devuelve un diccionario con la letra y su confianza si supera el umbral, o un diccionario vacío si no se alcanza la confianza mínima
            
        except Exception as e:
            return {}

def cargar_letras():
    return {}