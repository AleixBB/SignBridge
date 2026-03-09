import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# NOMBRES DE FEATURES
NOMBRES_FEATURES = [
    "flex_index_1","flex_index_2",
    "flex_middle_1","flex_middle_2",
    "flex_ring_1","flex_ring_2",
    "flex_pinky_1","flex_pinky_2",
    "flex_thumb",
    "dist_index_middle",
    "dist_middle_ring",
    "dist_ring_pinky",
    "dist_thumb_index",
    "dist_thumb_middle",
    "ext_index",
    "ext_middle",
    "ext_ring",
    "ext_pinky",
    "ext_thumb",
    "palm_orientation_x",
    "palm_orientation_y",
    "palm_orientation_z"
]
# PESOS DE FEATURES
PESOS_FEATURES = np.array([
    1,1,   # flex_index
    1,1,   # flex_middle
    1,1,   # flex_ring
    1,1,   # flex_pinky
    1,     # flex_thumb
    1,     # dist_index_middle
    1,     # dist_middle_ring
    1,     # dist_ring_pinky
    1,     # dist_thumb_index
    1,     # dist_thumb_middle
    1,     # ext_index
    1,     # ext_middle
    1,     # ext_ring
    1,     # ext_pinky
    1,     # ext_thumb
    0.5,   # palm_orientation_x
    0.5,   # palm_orientation_y
    0.2    # palm_orientation_z  
])

# ==========================================
# ANGULO ENTRE 3 PUNTOS
# ==========================================
def angulo(a,b,c):
    ba = a-b
    bc = c-b
    cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    cosang = np.clip(cosang,-1,1)
    return np.arccos(cosang)

# ==========================================
# EXTRAER FEATURES
# ==========================================
def extraer_features(vector):
    puntos = np.array(vector).reshape(21,3) #21 puntos con 3 coordenadas cada uno (x,y,z)
    wrist = puntos[0] 
    puntos = puntos - wrist #centrar al canell
    escala = np.linalg.norm(puntos[9]) #distancia del dedo medio al canell com a referencia de escala
    if escala != 0: #evitar divisió per zero
        puntos = puntos / escala 

    # FLEXION DEDOS
    dedos = [(5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)] #agrupamos los puntos de cada dedo: (mcp,pip,dip,tip)
    flexiones = []
    for mcp,pip,dip,tip in dedos: #per cada dedo, calculamos los ángulos de flexión en las articulaciones
        ang1 = angulo(puntos[mcp],puntos[pip],puntos[dip])
        ang2 = angulo(puntos[pip],puntos[dip],puntos[tip])
        flexiones.append(ang1)
        flexiones.append(ang2)
    ang_thumb = angulo(puntos[2],puntos[3],puntos[4])
    flexiones.append(ang_thumb)

    # DISTANCIAS
    distancias = [
        np.linalg.norm(puntos[8]-puntos[12]),
        np.linalg.norm(puntos[12]-puntos[16]),
        np.linalg.norm(puntos[16]-puntos[20]),
        np.linalg.norm(puntos[4]-puntos[8]),
        np.linalg.norm(puntos[4]-puntos[12])
    ]

    # EXTENSION
    extensiones = [
        np.linalg.norm(puntos[8]),
        np.linalg.norm(puntos[12]),
        np.linalg.norm(puntos[16]),
        np.linalg.norm(puntos[20]),
        np.linalg.norm(puntos[4])
    ]

    # ORIENTACION PALMA
    v1 = puntos[5]-puntos[0]
    v2 = puntos[17]-puntos[0]
    normal = np.cross(v1,v2)
    orientacion = normal/(np.linalg.norm(normal)+1e-6)

    features = np.concatenate([flexiones, distancias, extensiones, orientacion])
    return features

# ==========================================
# CARGAR DATASET
# ==========================================
def cargar_letras():
    letras = {}
    with open('/Users/aleixbenet/Desktop/datasetProva.csv','r',encoding='utf-8-sig') as f:
        for linea in f:
            linea = linea.strip() #strip el que fa és eliminar els espais en blanc al principi i al final de la línia, així com els caràcters de nova línia.
            #Això és útil per assegurar-se que no hi hagi caràcters no desitjats que puguin afectar el processament de les dades.
            if not linea:
                continue
            letra = linea[0]
            numeros = [float(x) for x in linea[1:].strip(',').split(',')] #aquí separem la línia en parts: la primera part és la lletra i la resta
            #coordenades. L'strip(',') elimina qualsevol coma extra al final de la cadena abans de dividir-la en números.
            if len(numeros) == 63:
                features = extraer_features(numeros)
                if letra not in letras:
                    letras[letra] = []
                letras[letra].append(features)
    return letras

# ==========================================
# COMPARADOR CON PESOS
# ==========================================
class Comparador:
    def __init__(self,dataset):
        self.refs = {}
        for letra,muestras in dataset.items():
            self.refs[letra] = np.mean(muestras,axis=0)

    def comparar(self,vector):
        features = extraer_features(vector)
        resultados = {}
        errores = {}
        confianza_features = {}
        for letra,ref in self.refs.items(): #comparamos las features extraídas con la referencia de cada letra
            # aplicar pesos
            dif = np.abs(features-ref) * PESOS_FEATURES #calculamos la diferencia entre las features extraídas y las de referencia, multiplicada por los pesos para dar más importancia a ciertas características
            dist = np.linalg.norm(dif)
            sim = max(0,100-dist*40)
            resultados[letra] = sim

            # confianza por feature
            confs = [max(0,100-d*100) for d in dif]
            confianza_features[letra] = confs

            # característica que falla más
            peor_idx = np.argmax(dif)
            errores[letra] = (NOMBRES_FEATURES[peor_idx], dif[peor_idx])
        return resultados, errores, confianza_features

# ==========================================
# INICIO
# ==========================================
print("Cargando dataset...")
dataset = cargar_letras()
if not dataset:
    print("❌ No se pudieron cargar letras")
    exit()
print("Letras disponibles:",list(dataset.keys()))
comparador = Comparador(dataset)

# ==========================================
# MEDIAPIPE
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,max_num_hands=1,
                       min_detection_confidence=0.7,min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
buffer = deque(maxlen=9) #buffer para suavizar la letra mostrada, se muestra la letra más frecuente en las últimas 6 frames

# ==========================================
# LOOP
# ==========================================
while True:
    ret,frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)
            vector = np.array([lm.x for lm in hand.landmark] + [lm.y for lm in hand.landmark] + [lm.z for lm in hand.landmark])
            vector = np.array([lm for lm in sum([[lm.x,lm.y,lm.z] for lm in hand.landmark],[])])
            
            resultados,errores,confianza_features = comparador.comparar(vector)
            mejor = max(resultados,key=resultados.get)
            buffer.append(mejor)
            letra = max(set(buffer),key=buffer.count)

            # LETRA
            cv2.putText(frame,f"Letra: {letra}",(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            # CONFIANZA GLOBAL
            conf = resultados[mejor]
            cv2.putText(frame,f"Confianza: {conf:.1f}%",(20,100),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            # ERROR PRINCIPAL
            caracteristica,_ = errores[mejor]
            cv2.putText(frame,f"Error principal: {caracteristica}",(20,130),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
            # CONFIANZA DE FEATURES
            y=170
            for i,nombre in enumerate(NOMBRES_FEATURES):
                conf = confianza_features[mejor][i]
                cv2.putText(frame,f"{nombre}: {conf:.0f}%",(20,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
                y += 18

    cv2.imshow("Detector de Letras",frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()